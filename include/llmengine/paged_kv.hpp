#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <mutex>
#include <vector>

namespace llmengine {

// Pool-allocated KV storage for the paged attention path.
//
// One block_id maps to storage for `block_size` tokens across ALL layers
// (one block_id covers the whole layer stack, matching the v6 plan §6.1).
// Per-block layout, FP32:
//   [K layer 0 (block_size * num_kv * hd floats)]
//   [K layer 1] ... [K layer L-1]
//   [V layer 0] ... [V layer L-1]
class BlockManager {
public:
    BlockManager(int num_blocks,
                 int block_size,
                 int num_layers,
                 int num_kv_heads,
                 int head_dim);

    int  allocate();                    // returns block_id, or -1 if exhausted
    void free(int block_id);
    std::size_t free_blocks() const;

    int num_blocks()  const { return num_blocks_; }
    int block_size()  const { return block_size_; }
    int num_layers()  const { return num_layers_; }
    int num_kv_heads() const { return num_kv_heads_; }
    int head_dim()    const { return head_dim_; }

    float*       k_ptr(int block_id, int layer);
    float*       v_ptr(int block_id, int layer);
    const float* k_ptr(int block_id, int layer) const;
    const float* v_ptr(int block_id, int layer) const;

private:
    int num_blocks_, block_size_, num_layers_, num_kv_heads_, head_dim_;
    std::size_t per_token_;             // num_kv_heads * head_dim floats
    std::size_t per_layer_per_block_;   // block_size * per_token (floats)
    std::size_t per_block_;             // 2 * num_layers * per_layer_per_block (floats: K + V)
    std::vector<float> pool_;
    std::list<int> free_list_;
    mutable std::mutex mu_;
};

// Per-sequence paged KV cache. Holds a list of block_ids that grow as the
// sequence is written, one new allocation per block_size tokens.
class PagedKVCache {
public:
    explicit PagedKVCache(BlockManager& mgr);
    ~PagedKVCache() { release_all(); }

    PagedKVCache(const PagedKVCache&) = delete;
    PagedKVCache& operator=(const PagedKVCache&) = delete;
    PagedKVCache(PagedKVCache&&) noexcept;
    PagedKVCache& operator=(PagedKVCache&&) noexcept;

    // Write k/v for a single token at (layer, pos). Allocates a new block
    // when pos crosses a block boundary. Returns false on pool exhaustion.
    // The caller terminates the seq with finish_reason="capacity" and runs
    // release_all() on terminal transitions.
    [[nodiscard]] bool write(int layer, int pos, const float* k, const float* v);

    // Gather positions [0, max_pos) of `layer` into the contiguous output
    // buffers. Each output buffer has shape [max_pos, num_kv_heads, head_dim].
    void gather(int layer, int max_pos, float* k_out, float* v_out) const;

    // Return every owned block_id to the manager's free list and clear
    // the table. Idempotent.
    void release_all();

    int  num_blocks_in_use() const { return static_cast<int>(block_table_.size()); }
    const std::vector<int>& block_table() const { return block_table_; }

private:
    BlockManager* mgr_;
    std::vector<int> block_table_;
};

}  // namespace llmengine
