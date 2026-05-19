#include "llmengine/scheduler.hpp"

#include <algorithm>
#include <stdexcept>

namespace llmengine {

namespace {

int argmax_logits(const float* logits, int n) {
    int best = 0;
    float bv = logits[0];
    for (int i = 1; i < n; ++i)
        if (logits[i] > bv) { bv = logits[i]; best = i; }
    return best;
}

SequenceResult snapshot(const Sequence& s) {
    return {s.seq_id, s.prompt_len, s.token_ids, s.finish_reason};
}

}  // namespace

// ---------- StaticBatchScheduler ----------

StaticBatchScheduler::StaticBatchScheduler(Engine& engine, BlockManager& mgr)
    : engine_(engine), mgr_(mgr) {}

void StaticBatchScheduler::enqueue(std::vector<std::int32_t> prompt,
                                   int max_new_tokens) {
    // Match engine.generate's input contract so the two paths agree.
    if (prompt.empty())
        throw std::invalid_argument("enqueue: prompt is empty");
    if (max_new_tokens < 0)
        throw std::invalid_argument("enqueue: max_new_tokens negative");
    const int V = engine_.config().vocab_size;
    for (auto t : prompt)
        if (t < 0 || t >= V)
            throw std::invalid_argument(
                "enqueue: token id " + std::to_string(t)
                + " out of range [0, " + std::to_string(V) + ")");
    // int64 sum so max_new_tokens near INT_MAX can't wrap past the guard.
    if (static_cast<std::int64_t>(prompt.size()) + max_new_tokens
        > engine_.max_pos())
        throw std::invalid_argument(
            "enqueue: prompt + max_new_tokens exceeds RoPE max_pos");

    auto s = std::make_unique<Sequence>(next_id_++, std::move(prompt),
                                        max_new_tokens, mgr_);
    s->state = SeqState::WAITING;
    seqs_.push_back(std::move(s));
}

void StaticBatchScheduler::run_until_done() {
    const auto& cfg = engine_.config();
    const int V = cfg.vocab_size;
    std::vector<float> logits(V);

    // Prefill every seq sequentially.
    for (auto& s : seqs_) {
        s->state = SeqState::PREFILLING;
        for (int i = 0; i < s->prompt_len; ++i) {
            const bool last = (i + 1 == s->prompt_len);
            bool ok = engine_.forward_step_paged(
                s->token_ids[i], i, *s->kv, last ? logits.data() : nullptr);
            if (!ok) { s->state = SeqState::FINISHED; s->finish_reason = "capacity"; break; }
        }
        if (s->state == SeqState::FINISHED) continue;
        // Match engine.generate's loop bound semantics: when max_new_tokens
        // is 0 we don't sample at all (the while-condition is false), so
        // finish_reason stays "length" without an EOS check. Otherwise
        // sample the first decoded token from the last prefill's logits;
        // the cap check happens RIGHT AFTER the push so max_new_tokens=1
        // produces exactly one generated token.
        if (s->max_new_tokens == 0) {
            s->state = SeqState::FINISHED;
            s->finish_reason = "length";
            continue;
        }
        int next = argmax_logits(logits.data(), V);
        if (cfg.eos_token_ids.count(next)) {
            s->state = SeqState::FINISHED;
            s->finish_reason = "stop";
            continue;
        }
        s->token_ids.push_back(next);
        if (static_cast<int>(s->token_ids.size()) - s->prompt_len
            >= s->max_new_tokens) {
            s->state = SeqState::FINISHED;
            s->finish_reason = "length";
        } else {
            s->state = SeqState::RUNNING;
        }
    }

    // Decode in lockstep.
    while (true) {
        bool any_running = false;
        for (auto& s : seqs_) {
            if (s->state != SeqState::RUNNING) continue;
            any_running = true;
            const int pos = static_cast<int>(s->token_ids.size()) - 1;
            bool ok = engine_.forward_step_paged(
                s->token_ids[pos], pos, *s->kv, logits.data());
            if (!ok) {
                s->state = SeqState::FINISHED;
                s->finish_reason = "capacity";
                continue;
            }
            int next = argmax_logits(logits.data(), V);
            if (cfg.eos_token_ids.count(next)) {
                s->state = SeqState::FINISHED;
                s->finish_reason = "stop";
                continue;
            }
            s->token_ids.push_back(next);
            if (static_cast<int>(s->token_ids.size()) - s->prompt_len
                >= s->max_new_tokens) {
                s->state = SeqState::FINISHED;
                s->finish_reason = "length";
            }
        }
        if (!any_running) break;
    }

    // Drain blocks, snapshot results.
    for (auto& s : seqs_) {
        s->kv->release_all();
        results_.push_back(snapshot(*s));
    }
    seqs_.clear();
}

// ---------- ContinuousBatchScheduler ----------

ContinuousBatchScheduler::ContinuousBatchScheduler(
    Engine& engine, BlockManager& mgr,
    int max_concurrent, int budget)
    : engine_(engine), mgr_(mgr),
      max_concurrent_(max_concurrent),
      budget_(budget) {
    // Without these guards run_until_idle() spins forever: max_concurrent<=0
    // blocks every admission and budget<=0 blocks every prefill advance, so
    // step() returns "not idle" indefinitely while making no progress.
    if (max_concurrent_ <= 0)
        throw std::invalid_argument(
            "ContinuousBatchScheduler: max_concurrent must be >= 1");
    if (budget_ <= 0)
        throw std::invalid_argument(
            "ContinuousBatchScheduler: max_prefill_tokens_per_step must be >= 1");
}

void ContinuousBatchScheduler::enqueue(std::vector<std::int32_t> prompt,
                                      int max_new_tokens) {
    // Match engine.generate's input contract so the two paths agree.
    if (prompt.empty())
        throw std::invalid_argument("enqueue: prompt is empty");
    if (max_new_tokens < 0)
        throw std::invalid_argument("enqueue: max_new_tokens negative");
    const int V = engine_.config().vocab_size;
    for (auto t : prompt)
        if (t < 0 || t >= V)
            throw std::invalid_argument(
                "enqueue: token id " + std::to_string(t)
                + " out of range [0, " + std::to_string(V) + ")");
    if (static_cast<std::int64_t>(prompt.size()) + max_new_tokens
        > engine_.max_pos())
        throw std::invalid_argument(
            "enqueue: prompt + max_new_tokens exceeds RoPE max_pos");

    auto s = std::make_unique<Sequence>(next_id_++, std::move(prompt),
                                        max_new_tokens, mgr_);
    s->state = SeqState::WAITING;
    waiting_.push(std::move(s));
}

void ContinuousBatchScheduler::advance_prefilling() {
    const int V = engine_.config().vocab_size;
    std::vector<float> logits(V);
    int budget = budget_;

    auto it = prefilling_.begin();
    while (it != prefilling_.end() && budget > 0) {
        auto& s = **it;
        while (budget > 0 && s.prefill_cursor < s.prompt_len) {
            const bool last = (s.prefill_cursor + 1 == s.prompt_len);
            bool ok = engine_.forward_step_paged(
                s.token_ids[s.prefill_cursor], s.prefill_cursor, *s.kv,
                last ? logits.data() : nullptr);
            if (!ok) {
                s.state = SeqState::FINISHED;
                s.finish_reason = "capacity";
                break;
            }
            s.prefill_cursor++;
            budget--;
        }

        if (s.state == SeqState::FINISHED) { ++it; continue; }

        if (s.prefill_cursor == s.prompt_len) {
            // engine.generate skips sampling entirely when max_new_tokens=0;
            // mirror that here so finish_reason matches across paths.
            if (s.max_new_tokens == 0) {
                s.state = SeqState::FINISHED;
                s.finish_reason = "length";
                ++it; continue;
            }
            int next = argmax_logits(logits.data(), V);
            if (engine_.config().eos_token_ids.count(next)) {
                s.state = SeqState::FINISHED;
                s.finish_reason = "stop";
                ++it; continue;
            }
            s.token_ids.push_back(next);
            // Check the cap RIGHT AFTER the push so max_new_tokens=1 produces
            // exactly one generated token. Without this the decode loop runs
            // one extra time before its own cap check fires.
            if (static_cast<int>(s.token_ids.size()) - s.prompt_len
                >= s.max_new_tokens) {
                s.state = SeqState::FINISHED;
                s.finish_reason = "length";
                ++it; continue;
            }
            s.state = SeqState::RUNNING;
            running_.push_back(std::move(*it));
            it = prefilling_.erase(it);
        } else {
            // Still PREFILLING; budget exhausted for this step.
            ++it;
        }
    }
}

void ContinuousBatchScheduler::decode_running() {
    const auto& cfg = engine_.config();
    const int V = cfg.vocab_size;
    std::vector<float> logits(V);

    for (auto& s : running_) {
        if (s->state != SeqState::RUNNING) continue;
        const int pos = static_cast<int>(s->token_ids.size()) - 1;
        bool ok = engine_.forward_step_paged(
            s->token_ids[pos], pos, *s->kv, logits.data());
        if (!ok) {
            s->state = SeqState::FINISHED;
            s->finish_reason = "capacity";
            continue;
        }
        int next = argmax_logits(logits.data(), V);
        if (cfg.eos_token_ids.count(next)) {
            s->state = SeqState::FINISHED;
            s->finish_reason = "stop";
            continue;
        }
        s->token_ids.push_back(next);
        if (static_cast<int>(s->token_ids.size()) - s->prompt_len
            >= s->max_new_tokens) {
            s->state = SeqState::FINISHED;
            s->finish_reason = "length";
        }
    }
}

void ContinuousBatchScheduler::finalize_finished(
    std::vector<std::unique_ptr<Sequence>>& list) {
    auto new_end = std::partition(list.begin(), list.end(),
        [](const auto& s) { return s->state != SeqState::FINISHED; });
    for (auto it = new_end; it != list.end(); ++it) {
        (*it)->kv->release_all();
        results_.push_back(snapshot(**it));
    }
    list.erase(new_end, list.end());
}

bool ContinuousBatchScheduler::step() {
    finalize_finished(prefilling_);
    finalize_finished(running_);

    while (!waiting_.empty()
           && static_cast<int>(running_.size() + prefilling_.size()) < max_concurrent_) {
        auto s = std::move(waiting_.front());
        waiting_.pop();
        s->state = SeqState::PREFILLING;
        prefilling_.push_back(std::move(s));
    }

    advance_prefilling();
    decode_running();

    return !idle();
}

void ContinuousBatchScheduler::run_until_idle() {
    while (step()) { /* loop */ }
    finalize_finished(prefilling_);
    finalize_finished(running_);
}

}  // namespace llmengine
