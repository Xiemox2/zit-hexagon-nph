#ifndef __SD_MODEL_VAE_VAE_HPP__
#define __SD_MODEL_VAE_VAE_HPP__

#include "core/tensor_ggml.hpp"
#include "model/common/block.hpp"
#include "model/vae/vae_tiling.hpp"
#include "model_manager.h"
#include "runtime/tiling.h"
#include "nph_stats.h"
#include "nph_npg_part.h"
#include <cmath>
#include <cstdlib>

struct VAE : public GGMLRunner {
protected:
    SDVersion version;
    std::string weight_prefix;
    bool scale_input                                      = true;
    virtual sd::Tensor<float> _compute(const int n_threads,
                                       const sd::Tensor<float>& z,
                                       bool decode_graph) = 0;

    virtual bool supports_temporal_tiling(VAETemporalDirection direction) const {
        SD_UNUSED(direction);
        return false;
    }

    virtual int get_default_temporal_tile_frames(VAETemporalDirection direction) const {
        SD_UNUSED(direction);
        return 4;
    }

    virtual int get_default_temporal_tile_overlap(VAETemporalDirection direction) const {
        SD_UNUSED(direction);
        return 1;
    }

    virtual int get_temporal_tile_output_scale(VAETemporalDirection direction) const {
        SD_UNUSED(direction);
        return 1;
    }

    virtual sd::Tensor<float> _compute_temporal_tiled(const int n_threads,
                                                      const sd::Tensor<float>& input,
                                                      VAETemporalDirection direction,
                                                      const VAETemporalTilingConfig& config) {
        if (direction != VAETemporalDirection::DECODE) {
            return _compute(n_threads, input, false);
        }

        VAETemporalTilingConfig resolved_config = config;
        const int output_scale                  = get_temporal_tile_output_scale(direction);
        if (output_scale > 1 &&
            resolved_config.overlap == 0 &&
            input.shape()[2] > resolved_config.tile_frames) {
            LOG_WARN("%s temporal decode requires at least one overlapping latent frame; using overlap=1",
                     get_desc().c_str());
            resolved_config.overlap = 1;
        }

        auto plan = make_vae_temporal_tile_plan(input.shape()[2], resolved_config);
        LOG_VERBOSE("%s temporal tiling: tile_frames=%d, overlap=%d, total_frames=%lld, tiles=%d",
                    get_desc().c_str(),
                    plan.tile_frames,
                    plan.overlap,
                    (long long)input.shape()[2],
                    (int)plan.tiles.size());
        return process_vae_temporal_tiles_blended(
            input,
            plan,
            output_scale,
            [&](const sd::Tensor<float>& input_tile, const VAETemporalTile& tile) {
                LOG_VERBOSE("%s temporal tile %d/%d: input frames [%lld, %lld)",
                            get_desc().c_str(),
                            tile.index + 1,
                            (int)plan.tiles.size(),
                            (long long)tile.start,
                            (long long)tile.end);
                return _compute(n_threads, input_tile, true);
            });
    }

    sd::Tensor<float> compute_with_temporal_tiling(const int n_threads,
                                                   const sd::Tensor<float>& input,
                                                   VAETemporalDirection direction,
                                                   const sd_tiling_params_t& tiling_params) {
        if (!tiling_params.temporal_tiling || input.dim() != 5 || input.shape()[2] <= 1) {
            return _compute(n_threads, input, direction == VAETemporalDirection::DECODE);
        }
        if (!supports_temporal_tiling(direction)) {
            LOG_WARN("%s does not support temporal tiling for %s; processing the full temporal dimension",
                     get_desc().c_str(),
                     direction == VAETemporalDirection::DECODE ? "decode" : "encode");
            return _compute(n_threads, input, direction == VAETemporalDirection::DECODE);
        }

        auto config = resolve_vae_temporal_tiling_config(
            tiling_params,
            get_default_temporal_tile_frames(direction),
            get_default_temporal_tile_overlap(direction));
        return _compute_temporal_tiled(n_threads, input, direction, config);
    }

    static inline void scale_tensor_to_minus1_1(sd::Tensor<float>* tensor) {
        GGML_ASSERT(tensor != nullptr);
        for (int64_t i = 0; i < tensor->numel(); ++i) {
            (*tensor)[i] = (*tensor)[i] * 2.0f - 1.0f;
        }
    }

    static inline void scale_tensor_to_0_1(sd::Tensor<float>* tensor) {
        GGML_ASSERT(tensor != nullptr);
        for (int64_t i = 0; i < tensor->numel(); ++i) {
            float value  = ((*tensor)[i] + 1.0f) * 0.5f;
            (*tensor)[i] = std::max(0.0f, std::min(1.0f, value));
        }
    }

    sd::Tensor<float> tiled_compute(const sd::Tensor<float>& input,
                                    int n_threads,
                                    int output_width,
                                    int output_height,
                                    int scale,
                                    int p_tile_size_x,
                                    int p_tile_size_y,
                                    float tile_overlap_factor,
                                    bool circular_x,
                                    bool circular_y,
                                    bool decode_graph,
                                    const sd_tiling_params_t& tiling_params,
                                    const char* error_message,
                                    bool silent = false) {
        auto on_processing = [&](const sd::Tensor<float>& input_tile) {
            auto output_tile = nph_compute_part(
                n_threads,
                input_tile,
                decode_graph ? VAETemporalDirection::DECODE : VAETemporalDirection::ENCODE,
                tiling_params);
            if (output_tile.empty()) {
                LOG_ERROR("%s", error_message);
                return sd::Tensor<float>();
            }
            return output_tile;
        };
        return ::process_tiles_2d(input,
                                  output_width,
                                  output_height,
                                  scale,
                                  p_tile_size_x,
                                  p_tile_size_y,
                                  tile_overlap_factor,
                                  circular_x,
                                  circular_y,
                                  on_processing,
                                  silent);
    }

    // ---- NPH 0.30 NPG: per-part check + NPU redo (experimental; only when nph_part_check_ is on) ----
    // A "part" is one decode tile (VAE tiling on) or the whole latent (tiling off). Each GPU part is
    // checked for an OpenCL/compute error (empty result), non-finite values and overflow (|v| above
    // NPH_NPG_PART_MAX, default 64: decoder output is ~[-1, 1]). Only the failing part is recomputed
    // with every node pinned to the NPU assist backend; healthy parts keep the GPU result.
    bool nph_part_check_ = false;
    int nph_parts_ = 0, nph_parts_redone_ = 0, nph_parts_failed_ = 0;
    std::string nph_parts_detail_;

    static const char* nph_part_problem(const sd::Tensor<float>& t, float limit, char* buf, size_t cap) {
        return t.empty() ? nph_npg_part_problem(nullptr, 0, limit, buf, cap)
                         : nph_npg_part_problem(t.data(), t.numel(), limit, buf, cap);
    }

    sd::Tensor<float> nph_compute_part(const int n_threads,
                                       const sd::Tensor<float>& input,
                                       VAETemporalDirection direction,
                                       const sd_tiling_params_t& tiling_params) {
        auto out = compute_with_temporal_tiling(n_threads, input, direction, tiling_params);
        if (!nph_part_check_ || direction != VAETemporalDirection::DECODE || nph_assist_backend_ == nullptr) {
            return out;
        }
        const int idx = ++nph_parts_;
        static const float limit = [] {
            const char* e = getenv("NPH_NPG_PART_MAX");
            const float v = e ? (float)atof(e) : 64.f;
            return v > 1.f ? v : 64.f;
        }();
        char why[160];
        const char* bad = nph_part_problem(out, limit, why, sizeof why);
        if (bad == nullptr) return out;
        const std::string bad_s = bad;
        const int64_t t0 = ggml_time_ms();
        nph_set_force_backend(nph_assist_backend_);
        auto redo = compute_with_temporal_tiling(n_threads, input, direction, tiling_params);
        nph_set_force_backend(nullptr);
        char why2[160];
        const char* bad2 = nph_part_problem(redo, limit, why2, sizeof why2);
        char line[400];
        if (bad2 == nullptr) {
            ++nph_parts_redone_;
            snprintf(line, sizeof line, "parte %d: GPU %s → rehecha en NPU OK (%lld ms)", idx, bad_s.c_str(),
                     (long long)(ggml_time_ms() - t0));
            out = std::move(redo);
        } else {
            ++nph_parts_failed_;
            snprintf(line, sizeof line, "parte %d: GPU %s → NPU también falló (%s)", idx, bad_s.c_str(), bad2);
            out = sd::Tensor<float>();  // let the caller fail → app-level NPU redo of the whole image
        }
        LOG_WARN("nph-npg-part: %s", line);
        if (nph_parts_redone_ + nph_parts_failed_ <= 8) {
            nph_parts_detail_ += line;
            nph_parts_detail_ += "\n";
        }
        return out;
    }

    void nph_parts_summary() {
        if (!nph_part_check_ || nph_parts_ == 0) return;
        char v[200];
        snprintf(v, sizeof v, "%d partes en %s · rehechas en NPU %d · fallidas %d", nph_parts_,
                 ggml_backend_name(runtime_backend), nph_parts_redone_, nph_parts_failed_);
        LOG_INFO("nph-npg-parts: %s", v);
        nph_npg_note("partes VAE", v);
        if (!nph_parts_detail_.empty()) nph_npg_note("partes rehechas", nph_parts_detail_.c_str());
        nph_parts_ = nph_parts_redone_ = nph_parts_failed_ = 0;
        nph_parts_detail_.clear();
    }

public:
    void nph_enable_part_check(bool on) { nph_part_check_ = on; }

    VAE(SDVersion version,
        ggml_backend_t backend,
        const std::string& weight_prefix                    = "",
        std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : version(version), weight_prefix(weight_prefix), GGMLRunner(backend, weight_manager) {}

    int get_scale_factor() {
        int scale_factor = 8;
        if (version == VERSION_LTXAV) {
            scale_factor = 32;
        } else if (version == VERSION_WAN2_2_TI2V || version == VERSION_QWEN_IMAGE_2_1 || sd_version_is_hunyuan_video(version) || sd_version_is_mage_flow(version) || sd_version_is_minimax_h3(version)) {
            scale_factor = 16;
        } else if (sd_version_uses_flux2_vae(version)) {
            scale_factor = 16;
        } else if (version == VERSION_CHROMA_RADIANCE || version == VERSION_HIDREAM_O1 || sd_version_is_minit2i(version)) {
            scale_factor = 1;
        }
        return scale_factor;
    }

    virtual int get_encoder_output_channels(int input_channels) = 0;

    bool can_temporal_tile_decode() const {
        return supports_temporal_tiling(VAETemporalDirection::DECODE);
    }

    void get_tile_sizes(int& tile_size_x,
                        int& tile_size_y,
                        float& tile_overlap,
                        const sd_tiling_params_t& params,
                        int64_t latent_x,
                        int64_t latent_y,
                        float encoding_factor = 1.0f) {
        tile_overlap       = std::max(std::min(params.target_overlap, 0.5f), 0.0f);
        auto get_tile_size = [&](int requested_size, float factor, int64_t latent_size) {
            const int default_tile_size  = 32;
            const int min_tile_dimension = 4;
            int tile_size                = default_tile_size;
            // factor <= 1 means simple fraction of the latent dimension
            // factor > 1 means number of tiles across that dimension
            if (factor > 0.f) {
                if (factor > 1.0)
                    factor = 1 / (factor - factor * tile_overlap + tile_overlap);
                tile_size = static_cast<int>(std::round(latent_size * factor));
            } else if (requested_size >= min_tile_dimension) {
                tile_size = requested_size;
            }
            tile_size = static_cast<int>(tile_size * encoding_factor);
            return std::max(std::min(tile_size, static_cast<int>(latent_size)), min_tile_dimension);
        };

        tile_size_x = get_tile_size(params.tile_size_x, params.rel_size_x, latent_x);
        tile_size_y = get_tile_size(params.tile_size_y, params.rel_size_y, latent_y);
    }

    virtual sd::Tensor<float> encode(int n_threads,
                                     const sd::Tensor<float>& x,
                                     sd_tiling_params_t tiling_params,
                                     bool circular_x = false,
                                     bool circular_y = false) {
        int64_t t0              = ggml_time_ms();
        sd::Tensor<float> input = x;
        sd::Tensor<float> output;
        if (scale_input) {
            scale_tensor_to_minus1_1(&input);
        }

        if (tiling_params.enabled) {
            const int scale_factor = get_scale_factor();
            int64_t W              = input.shape()[0] / scale_factor;
            int64_t H              = input.shape()[1] / scale_factor;
            float tile_overlap;
            int tile_size_x, tile_size_y;
            // Image VAE encode is more sensitive to tile boundary context than decode.
            // Keep the smaller legacy factor for video VAEs, but default image encode
            // tiles to 64 latent pixels so a 512px SD image is encoded as one tile.
            const float encode_tile_factor = sd_version_is_minimax_h3(version) ? 1.f : (sd_version_is_wan(version) || sd_version_is_hunyuan_video(version) || sd_version_is_ltxav(version)) ? 1.30539f
                                                                                                                                                                                            : 2.0f;
            get_tile_sizes(tile_size_x, tile_size_y, tile_overlap, tiling_params, W, H, encode_tile_factor);
            LOG_VERBOSE("VAE Tile size: %dx%d", tile_size_x, tile_size_y);
            output = tiled_compute(input,
                                   n_threads,
                                   static_cast<int>(W),
                                   static_cast<int>(H),
                                   scale_factor,
                                   tile_size_x,
                                   tile_size_y,
                                   tile_overlap,
                                   circular_x,
                                   circular_y,
                                   false,
                                   tiling_params,
                                   "vae encode compute failed while processing a tile");
        } else {
            output = compute_with_temporal_tiling(n_threads,
                                                  input,
                                                  VAETemporalDirection::ENCODE,
                                                  tiling_params);
        }

        runner_end();

        if (output.empty()) {
            LOG_ERROR("vae encode compute failed");
            return {};
        }
        int64_t t1 = ggml_time_ms();
        LOG_VERBOSE("computing vae encode graph completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);
        return std::move(output);
    }

    virtual sd::Tensor<float> decode(int n_threads,
                                     const sd::Tensor<float>& x,
                                     sd_tiling_params_t tiling_params,
                                     bool decode_video = false,
                                     bool circular_x   = false,
                                     bool circular_y   = false,
                                     bool silent       = false) {
        int64_t t0              = ggml_time_ms();
        sd::Tensor<float> input = x;
        sd::Tensor<float> output;
        if (nph_assist_backend_ != nullptr) {
            char v[200];
            snprintf(v, sizeof v, "VAE en %s · ayuda %s · mul_mat/atención→NPU %s · revisar/rehacer partes en NPU %s",
                     ggml_backend_name(runtime_backend), ggml_backend_name(nph_assist_backend_),
                     nph_mm_backend_ ? "sí" : "no", nph_part_check_ ? "sí" : "no");
            nph_npg_note("asistencia NPU", v);
        }

        if (tiling_params.enabled) {
            const int scale_factor = get_scale_factor();
            int64_t W              = input.shape()[0] * scale_factor;
            int64_t H              = input.shape()[1] * scale_factor;
            float tile_overlap;
            int tile_size_x, tile_size_y;
            get_tile_sizes(tile_size_x, tile_size_y, tile_overlap, tiling_params, input.shape()[0], input.shape()[1]);
            if (!silent) {
                LOG_VERBOSE("VAE Tile size: %dx%d", tile_size_x, tile_size_y);
            }
            output = tiled_compute(
                input,
                n_threads,
                static_cast<int>(W),
                static_cast<int>(H),
                scale_factor,
                tile_size_x,
                tile_size_y,
                tile_overlap,
                circular_x,
                circular_y,
                true,
                tiling_params,
                "vae decode compute failed while processing a tile",
                silent);
        } else {
            output = nph_compute_part(n_threads,
                                      input,
                                      VAETemporalDirection::DECODE,
                                      tiling_params);
        }

        runner_end();
        nph_parts_summary();

        if (output.empty()) {
            LOG_ERROR("vae decode compute failed");
            return {};
        }
        if (scale_input) {
            scale_tensor_to_0_1(&output);
        }
        int64_t t1 = ggml_time_ms();
        LOG_VERBOSE("computing vae decode graph completed, taking %.2fs", (t1 - t0) * 1.0f / 1000);
        return std::move(output);
    }

    virtual sd::Tensor<float> vae_output_to_latents(const sd::Tensor<float>& vae_output, std::shared_ptr<RNG> rng) = 0;
    virtual sd::Tensor<float> diffusion_to_vae_latents(const sd::Tensor<float>& latents)                           = 0;
    virtual sd::Tensor<float> vae_to_diffusion_latents(const sd::Tensor<float>& latents)                           = 0;
    virtual void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors)                                   = 0;
    virtual void set_conv2d_scale(float scale) { SD_UNUSED(scale); };
};

struct FakeVAE : public VAE {
    FakeVAE(SDVersion version,
            ggml_backend_t backend,
            std::shared_ptr<RunnerWeightManager> weight_manager = nullptr)
        : VAE(version, backend, "", weight_manager) {}

    int get_encoder_output_channels(int input_channels) {
        return input_channels;
    }

    sd::Tensor<float> _compute(const int n_threads,
                               const sd::Tensor<float>& z,
                               bool decode_graph) override {
        SD_UNUSED(n_threads);
        SD_UNUSED(decode_graph);
        return z;
    }

    sd::Tensor<float> vae_output_to_latents(const sd::Tensor<float>& vae_output, std::shared_ptr<RNG> rng) override {
        SD_UNUSED(rng);
        return vae_output;
    }

    sd::Tensor<float> diffusion_to_vae_latents(const sd::Tensor<float>& latents) override {
        return latents;
    }

    sd::Tensor<float> vae_to_diffusion_latents(const sd::Tensor<float>& latents) override {
        return latents;
    }

    void get_param_tensors(std::map<std::string, ggml_tensor*>& tensors) override {}

    std::string get_desc() override {
        return "fake_vae";
    }
};

#endif  // __SD_MODEL_VAE_VAE_HPP__
