#include "llmengine/paged_kv.hpp"

#include <cstring>
#include <stdexcept>

namespace llmengine {

BlockManager::BlockManager(int num_blocks,
                           int block_size,
                           int num_layers,
                           int num_kv_heads,
                           int head_dim)
    : num_blocks_(num_blocks),
      block_size_(block_size),
      num_layers_(num_layers),
      num_kv_heads_(num_kv_heads),
      head_dim_(head_dim) {
    if (num_blocks <= 0 || block_size <= 0 || num_layers <= 0
        || num_kv_heads <= 0 || head_dim <= 0)
        throw std::invalid_argument("BlockManager: non-positive dim");

    per_token_           = static_cast<std::size_t>(num_kv_heads) * head_dim;
    per_layer_per_block_ = static_cast<std::size_t>(block_size) * per_token_;
    per_block_           = 2ULL * num_layers * per_layer_per_block_;
    pool_.assign(static_cast<std::size_t>(num_blocks) * per_block_, 0.0f);

    for (int i = 0; i < num_blocks; ++i) free_list_.push_back(i);
}

int BlockManager::allocate() {
    std::lock_guard<std::mutex> lk(mu_);
    if (free_list_.empty()) return -1;
    int id = free_list_.front();
    free_list_.pop_front();
    return id;
}

void BlockManager::free(int id) {
    if (id < 0 || id >= num_blocks_) return;
    std::lock_guard<std::mutex> lk(mu_);
    free_list_.push_back(id);
}

std::size_t BlockManager::free_blocks() const {
    std::lock_guard<std::mutex> lk(mu_);
    return free_list_.size();
}

float* BlockManager::k_ptr(int block_id, int layer) {
    return &pool_[static_cast<std::size_t>(block_id) * per_block_
                  + static_cast<std::size_t>(layer) * per_layer_per_block_];
}
float* BlockManager::v_ptr(int block_id, int layer) {
    return &pool_[static_cast<std::size_t>(block_id) * per_block_
                  + (static_cast<std::size_t>(num_layers_) + layer)
                    * per_layer_per_block_];
}
const float* BlockManager::k_ptr(int block_id, int layer) const {
    return &pool_[static_cast<std::size_t>(block_id) * per_block_
                  + static_cast<std::size_t>(layer) * per_layer_per_block_];
}
const float* BlockManager::v_ptr(int block_id, int layer) const {
    return &pool_[static_cast<std::size_t>(block_id) * per_block_
                  + (static_cast<std::size_t>(num_layers_) + layer)
                    * per_layer_per_block_];
}

// ---- PagedKVCache ----

PagedKVCache::PagedKVCache(BlockManager& mgr) : mgr_(&mgr) {}

PagedKVCache::PagedKVCache(PagedKVCache&& o) noexcept
    : mgr_(o.mgr_), block_table_(std::move(o.block_table_)) {
    o.mgr_ = nullptr;
    o.block_table_.clear();
}
PagedKVCache& PagedKVCache::operator=(PagedKVCache&& o) noexcept {
    if (this != &o) {
        release_all();
        mgr_ = o.mgr_;
        block_table_ = std::move(o.block_table_);
        o.mgr_ = nullptr;
        o.block_table_.clear();
    }
    return *this;
}

bool PagedKVCache::write(int layer, int pos, const float* k, const float* v) {
    if (!mgr_) return false;
    const int bs = mgr_->block_size();
    const int blk_idx = pos / bs;
    const int blk_off = pos % bs;

    // Allocate when crossing into a new block. At pos=0 we always allocate
    // the first block on the first call; at pos=block_size we allocate the
    // second; etc.
    if (blk_idx >= static_cast<int>(block_table_.size())) {
        int new_id = mgr_->allocate();
        if (new_id < 0) return false;
        block_table_.push_back(new_id);
    }
    const int block_id = block_table_[blk_idx];

    const std::size_t per_tok = static_cast<std::size_t>(mgr_->num_kv_heads())
                              * mgr_->head_dim();
    std::memcpy(mgr_->k_ptr(block_id, layer) + blk_off * per_tok,
                k, per_tok * sizeof(float));
    std::memcpy(mgr_->v_ptr(block_id, layer) + blk_off * per_tok,
                v, per_tok * sizeof(float));
    return true;
}

void PagedKVCache::gather(int layer, int max_pos,
                          float* k_out, float* v_out) const {
    if (!mgr_) throw std::runtime_error("PagedKVCache::gather: empty cache");
    const int bs = mgr_->block_size();
    const std::size_t per_tok = static_cast<std::size_t>(mgr_->num_kv_heads())
                              * mgr_->head_dim();

    for (int p = 0; p < max_pos; ++p) {
        int blk_idx = p / bs;
        int blk_off = p % bs;
        if (blk_idx >= static_cast<int>(block_table_.size()))
            throw std::out_of_range("PagedKVCache::gather: pos beyond block_table");
        int block_id = block_table_[blk_idx];
        std::memcpy(k_out + static_cast<std::size_t>(p) * per_tok,
                    mgr_->k_ptr(block_id, layer) + blk_off * per_tok,
                    per_tok * sizeof(float));
        std::memcpy(v_out + static_cast<std::size_t>(p) * per_tok,
                    mgr_->v_ptr(block_id, layer) + blk_off * per_tok,
                    per_tok * sizeof(float));
    }
}

void PagedKVCache::release_all() {
    if (!mgr_) { block_table_.clear(); return; }
    for (int id : block_table_) mgr_->free(id);
    block_table_.clear();
}

}  // namespace llmengine
