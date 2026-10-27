// moe_probe: run prompts through a GGUF MoE model with llama.cpp and record, per layer,
// which experts the router selected (ffn_moe_topk) and the router probabilities (ffn_moe_probs).
//
//   moe_probe -m model.gguf -i prompts.jsonl -o routing.jsonl [-t threads] [-c n_ctx]
//
// prompts.jsonl: one {"idx": <int>, "text": <str>} per line.
// routing.jsonl: one line per prompt with counts[layer][expert] and probs[layer][expert]
//                summed over the prompt's tokens.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

using json = nlohmann::json;

static const int CHUNK = 128;

struct Routing {
    int top_k = 0, n_expert = 0;
    std::map<int, std::vector<int64_t>> counts; // layer -> [n_expert]
    std::map<int, std::vector<double>>  probs;  // layer -> [n_expert]
    void clear() { counts.clear(); probs.clear(); }
};

// Copy a 2-D tensor (possibly a strided view, e.g. the top-k slice of argsort) row by row.
template <typename T>
static std::vector<T> read_rows(const ggml_tensor * t) {
    const int64_t ne0 = t->ne[0], ne1 = t->ne[1];
    std::vector<T> out(ne0 * ne1);
    for (int64_t j = 0; j < ne1; ++j) {
        ggml_backend_tensor_get(t, out.data() + j * ne0, j * t->nb[1], ne0 * sizeof(T));
    }
    return out;
}

// Layer index if `name` is exactly "<prefix><digits>" (skips derived views like "ffn_moe_probs-3 (reshaped)"), else -1.
static int layer_of(const char * name, const char * prefix) {
    const size_t n = strlen(prefix);
    if (strncmp(name, prefix, n) != 0 || !name[n]) return -1;
    for (const char * c = name + n; *c; ++c) if (!isdigit((unsigned char) *c)) return -1;
    return atoi(name + n);
}

// Called by llama.cpp for every graph node: first with ask=true ("do you want it?"), then with the data.
static bool on_tensor(ggml_tensor * t, bool ask, void * user_data) {
    const int  il_topk  = layer_of(t->name, "ffn_moe_topk-");
    const int  il_probs = layer_of(t->name, "ffn_moe_probs-");
    const bool is_topk = il_topk >= 0, is_probs = il_probs >= 0;
    if (ask) return is_topk || is_probs;
    if (!is_topk && !is_probs) return true;

    auto & r = *static_cast<Routing *>(user_data);
    const int il = is_topk ? il_topk : il_probs;
    const int64_t ne0 = t->ne[0], n_tok = t->ne[1];

    if (is_probs) {                                   // F32 [n_expert, n_tokens]
        GGML_ASSERT(t->type == GGML_TYPE_F32);
        r.n_expert = ne0;
        auto & acc = r.probs[il];
        acc.resize(ne0, 0.0);
        auto p = read_rows<float>(t);
        for (int64_t j = 0; j < n_tok; ++j)
            for (int64_t e = 0; e < ne0; ++e) acc[e] += p[j * ne0 + e];
    } else {                                          // I32 [top_k, n_tokens]: selected expert ids
        GGML_ASSERT(t->type == GGML_TYPE_I32);
        r.top_k = ne0;
        auto ids = read_rows<int32_t>(t);
        auto & acc = r.counts[il];
        if (acc.empty()) acc.resize(r.n_expert, 0);   // probs node is computed before topk
        for (int32_t id : ids) acc.at(id)++;
    }
    return true;
}

static void quiet_log(ggml_log_level level, const char * text, void *) {
    if (level >= GGML_LOG_LEVEL_WARN) fputs(text, stderr);
}

int main(int argc, char ** argv) {
    std::string model_path, in_path, out_path;
    int n_threads = 4, n_ctx = 2048;
    for (int i = 1; i + 1 < argc; i += 2) {
        std::string a = argv[i];
        if      (a == "-m") model_path = argv[i + 1];
        else if (a == "-i") in_path    = argv[i + 1];
        else if (a == "-o") out_path   = argv[i + 1];
        else if (a == "-t") n_threads  = atoi(argv[i + 1]);
        else if (a == "-c") n_ctx      = atoi(argv[i + 1]);
    }
    if (model_path.empty() || in_path.empty() || out_path.empty()) {
        fprintf(stderr, "usage: %s -m model.gguf -i prompts.jsonl -o routing.jsonl [-t threads] [-c n_ctx]\n", argv[0]);
        return 1;
    }

    llama_log_set(quiet_log, nullptr);
    llama_backend_init();

    auto mparams = llama_model_default_params();
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mparams);
    if (!model) { fprintf(stderr, "failed to load %s\n", model_path.c_str()); return 1; }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    Routing routing;
    auto cparams = llama_context_default_params();
    cparams.n_ctx             = n_ctx;
    cparams.n_batch           = CHUNK;  // every token is an output (see below), so keep the logits buffer small
    cparams.n_ubatch          = CHUNK;
    cparams.n_threads         = n_threads;
    cparams.n_threads_batch   = n_threads;
    cparams.cb_eval           = on_tensor;
    cparams.cb_eval_user_data = &routing;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { fprintf(stderr, "failed to create context\n"); return 1; }

    std::ifstream fin(in_path);
    std::ofstream fout(out_path);
    std::string line;
    int done = 0;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        json in = json::parse(line);
        const std::string text = in["text"];

        std::vector<llama_token> toks(text.size() + 16);
        int n = llama_tokenize(vocab, text.c_str(), text.size(), toks.data(), toks.size(), true, false);
        if (n < 0) { toks.resize(-n); n = llama_tokenize(vocab, text.c_str(), text.size(), toks.data(), toks.size(), true, false); }
        toks.resize(std::min(n, n_ctx));

        routing.clear();
        llama_memory_clear(llama_get_memory(ctx), true);
        // Mark every token as an output: otherwise llama.cpp runs the last layer on the final token only.
        bool ok = true;
        for (size_t start = 0; ok && start < toks.size(); start += CHUNK) {
            const int n_chunk = std::min<size_t>(CHUNK, toks.size() - start);
            llama_batch batch = llama_batch_init(n_chunk, 0, 1);
            for (int j = 0; j < n_chunk; ++j) {
                batch.token[j] = toks[start + j]; batch.pos[j] = start + j;
                batch.n_seq_id[j] = 1; batch.seq_id[j][0] = 0; batch.logits[j] = true;
            }
            batch.n_tokens = n_chunk;
            ok = llama_decode(ctx, batch) == 0;
            llama_batch_free(batch);
        }
        if (!ok) { fprintf(stderr, "decode failed for idx %d\n", in["idx"].get<int>()); continue; }

        json out = {{"idx", in["idx"]}, {"n_tokens", toks.size()},
                    {"n_expert", routing.n_expert}, {"top_k", routing.top_k}};
        json layers = json::array(), counts = json::array(), probs = json::array();
        for (auto & [il, c] : routing.counts) {
            layers.push_back(il);
            counts.push_back(c);
            json p = json::array();
            for (double v : routing.probs[il]) p.push_back(std::round(v * 1e4) / 1e4);
            probs.push_back(p);
        }
        out["layers"] = layers; out["counts"] = counts; out["probs"] = probs;
        fout << out.dump() << "\n" << std::flush;
        fprintf(stderr, "[%d] idx=%d tokens=%zu moe_layers=%zu\n", ++done, in["idx"].get<int>(), toks.size(), layers.size());
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
