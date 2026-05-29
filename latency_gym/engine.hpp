#pragma once

#include <cstdint>
#include <vector>
#include <array>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <queue>
#include <random>

namespace latency_gym {

struct TimeCounter {
    uint64_t nanoseconds;
    
    TimeCounter() : nanoseconds(0) {}
    explicit TimeCounter(uint64_t ns) : nanoseconds(ns) {}
    
    TimeCounter operator+(const TimeCounter& other) const {
        return TimeCounter(nanoseconds + other.nanoseconds);
    }
    
    TimeCounter operator-(const TimeCounter& other) const {
        return TimeCounter(nanoseconds > other.nanoseconds ? 
                          nanoseconds - other.nanoseconds : 0);
    }
    
    bool operator<(const TimeCounter& other) const {
        return nanoseconds < other.nanoseconds;
    }
};

struct Order {
    uint64_t order_id;
    int64_t timestamp_ns;
    uint32_t size;
    bool is_buy;
    int64_t match_time_ns;
    
    Order() : order_id(0), timestamp_ns(0), size(0), is_buy(true), match_time_ns(-1) {}
    
    Order(uint64_t id, int64_t ts, uint32_t sz, bool buy) 
        : order_id(id), timestamp_ns(ts), size(sz), is_buy(buy), match_time_ns(-1) {}
};

class OrderRingBuffer {
private:
    std::vector<Order> buffer_;
    size_t write_idx_;
    size_t read_idx_;
    size_t count_;
    size_t capacity_;
    uint64_t total_overflow_;

public:
    explicit OrderRingBuffer(size_t capacity) 
        : buffer_(capacity), write_idx_(0), read_idx_(0), 
          count_(0), capacity_(capacity), total_overflow_(0) {}
    
    bool push(const Order& order) {
        if (count_ >= capacity_) {
            total_overflow_++;
            return false;
        }
        buffer_[write_idx_] = order;
        write_idx_ = (write_idx_ + 1) % capacity_;
        count_++;
        return true;
    }
    
    bool pop(Order& order) {
        if (count_ == 0) {
            return false;
        }
        order = buffer_[read_idx_];
        read_idx_ = (read_idx_ + 1) % capacity_;
        count_--;
        return true;
    }
    
    size_t size() const { return count_; }
    size_t capacity() const { return capacity_; }
    uint64_t overflow_count() const { return total_overflow_; }
    bool is_full() const { return count_ >= capacity_; }
    bool is_empty() const { return count_ == 0; }
    double fill_ratio() const { 
        return capacity_ > 0 ? static_cast<double>(count_) / capacity_ : 0.0; 
    }
};

class LatencyStatsWindow {
private:
    std::vector<uint64_t> latencies_;
    size_t max_size_;
    double rolling_mean_;
    double rolling_variance_;
    size_t count_;

public:
    explicit LatencyStatsWindow(size_t window_size = 1000)
        : max_size_(window_size), rolling_mean_(0.0), 
          rolling_variance_(0.0), count_(0) {
        latencies_.reserve(max_size_);
    }
    
    void record(uint64_t latency_ns) {
        if (latencies_.size() >= max_size_) {
            latencies_.erase(latencies_.begin());
        }
        latencies_.push_back(latency_ns);
        count_++;
        update_statistics();
    }
    
    double get_variance() const {
        return rolling_variance_;
    }
    
    double get_mean() const {
        return rolling_mean_;
    }
    
    uint64_t get_percentile(double p) const {
        if (latencies_.empty()) return 0;
        
        std::vector<uint64_t> sorted = latencies_;
        std::sort(sorted.begin(), sorted.end());
        
        size_t idx = static_cast<size_t>(
            (p / 100.0) * static_cast<double>(sorted.size() - 1)
        );
        return sorted[std::min(idx, sorted.size() - 1)];
    }
    
    size_t window_size() const { return latencies_.size(); }

private:
    void update_statistics() {
        if (latencies_.empty()) {
            rolling_mean_ = 0.0;
            rolling_variance_ = 0.0;
            return;
        }
        
        double sum = 0.0;
        for (uint64_t lat : latencies_) {
            sum += static_cast<double>(lat);
        }
        rolling_mean_ = sum / latencies_.size();
        
        double variance_sum = 0.0;
        for (uint64_t lat : latencies_) {
            double diff = static_cast<double>(lat) - rolling_mean_;
            variance_sum += diff * diff;
        }
        rolling_variance_ = variance_sum / latencies_.size();
    }
};

struct EnvironmentState {
    double queue_depth;
    uint64_t last_latency_ns;
    double latency_variance;
    uint64_t packet_drops;
    double queue_fill_ratio;
    uint64_t mean_latency_ns;
    uint64_t p99_latency_ns;
    uint64_t p999_latency_ns;
    
    EnvironmentState() 
        : queue_depth(0), last_latency_ns(0), latency_variance(0),
          packet_drops(0), queue_fill_ratio(0), mean_latency_ns(0),
          p99_latency_ns(0), p999_latency_ns(0) {}
};

class LatencySimulator {
private:
    uint32_t batch_size_;
    uint32_t polling_rate_;
    uint32_t prealloc_pool_;
    
    OrderRingBuffer order_buffer_;
    LatencyStatsWindow stats_window_;
    
    uint64_t current_time_ns_;
    uint64_t step_count_;
    std::mt19937_64 rng_;
    
    EnvironmentState current_state_;
    
    std::exponential_distribution<double> interarrival_dist_;
    std::uniform_int_distribution<uint32_t> order_size_dist_;
    std::bernoulli_distribution buy_sell_dist_;
    
    double last_reward_;
    uint64_t matched_orders_;
    uint64_t total_queue_time_ns_;

public:
    static constexpr size_t DEFAULT_BUFFER_CAPACITY = 4096;
    static constexpr uint64_t TIME_STEP_NS = 1000000;
    static constexpr uint32_t MIN_BATCH_SIZE = 1;
    static constexpr uint32_t MAX_BATCH_SIZE = 64;
    static constexpr uint32_t MIN_POLLING_RATE = 1;
    static constexpr uint32_t MAX_POLLING_RATE = 10;
    static constexpr uint32_t MIN_PREALLOC_POOL = 1;
    static constexpr uint32_t MAX_PREALLOC_POOL = 5;
    
    LatencySimulator(uint32_t seed = 42)
        : batch_size_(4), polling_rate_(5), prealloc_pool_(2),
          order_buffer_(DEFAULT_BUFFER_CAPACITY),
          stats_window_(1000),
          current_time_ns_(0), step_count_(0),
          rng_(seed),
          interarrival_dist_(0.1),
          order_size_dist_(100, 10000),
          buy_sell_dist_(0.5),
          last_reward_(0.0),
          matched_orders_(0),
          total_queue_time_ns_(0) {}
    
    void reset() {
        current_time_ns_ = 0;
        step_count_ = 0;
        matched_orders_ = 0;
        total_queue_time_ns_ = 0;
        last_reward_ = 0.0;
        
        Order dummy;
        while (order_buffer_.pop(dummy)) {}
        
        update_state();
    }
    
    void set_action(uint32_t batch_size, uint32_t polling_rate, uint32_t prealloc_pool) {
        batch_size_ = std::clamp(batch_size, MIN_BATCH_SIZE, MAX_BATCH_SIZE);
        polling_rate_ = std::clamp(polling_rate, MIN_POLLING_RATE, MAX_POLLING_RATE);
        prealloc_pool_ = std::clamp(prealloc_pool, MIN_PREALLOC_POOL, MAX_PREALLOC_POOL);
    }
    
    void step() {
        generate_synthetic_traffic();
        process_orders();
        current_time_ns_ += TIME_STEP_NS;
        step_count_++;
        update_state();
    }
    
    const EnvironmentState& get_state() const {
        return current_state_;
    }
    
    double compute_reward(double alpha = 1.0, double beta = 0.5, double gamma = 2.0) {
        double mean_penalty = alpha * static_cast<double>(current_state_.mean_latency_ns);
        double variance_penalty = beta * current_state_.latency_variance;
        double drop_penalty = gamma * static_cast<double>(current_state_.packet_drops);
        
        last_reward_ = -(mean_penalty + variance_penalty + drop_penalty);
        return last_reward_;
    }
    
    uint64_t get_step_count() const { return step_count_; }
    double get_last_reward() const { return last_reward_; }
    uint64_t get_matched_orders() const { return matched_orders_; }
    uint64_t get_total_overflow() const { return order_buffer_.overflow_count(); }

private:
    void generate_synthetic_traffic() {
        bool is_burst = (step_count_ % 10 == 0);
        uint32_t orders_to_generate = is_burst ? 32 : 8;
        
        for (uint32_t i = 0; i < orders_to_generate; ++i) {
            uint64_t arrival_time = current_time_ns_ + 
                static_cast<uint64_t>(interarrival_dist_(rng_) * 1000.0);
            
            uint32_t size = order_size_dist_(rng_);
            bool is_buy = buy_sell_dist_(rng_);
            
            Order new_order(
                matched_orders_ * 1000 + i,
                static_cast<int64_t>(arrival_time),
                size,
                is_buy
            );
            
            if (!order_buffer_.push(new_order)) {
                current_state_.packet_drops++;
            }
        }
    }
    
    void process_orders() {
        uint32_t processed = 0;
        uint32_t max_process = batch_size_;
        
        if (step_count_ % polling_rate_ != 0) {
            max_process = std::max(1U, batch_size_ / polling_rate_);
        }
        
        Order order;
        while (processed < max_process && order_buffer_.pop(order)) {
            if (order.match_time_ns < 0) {
                order.match_time_ns = static_cast<int64_t>(current_time_ns_);
            }
            
            uint64_t latency = order.match_time_ns > order.timestamp_ns ? 
                static_cast<uint64_t>(order.match_time_ns - order.timestamp_ns) : 1;
            
            latency = std::min(latency, static_cast<uint64_t>(1000000000));
            
            stats_window_.record(latency);
            total_queue_time_ns_ += latency;
            matched_orders_++;
            processed++;
        }
    }
    
    void update_state() {
        current_state_.queue_depth = static_cast<double>(order_buffer_.size());
        current_state_.queue_fill_ratio = order_buffer_.fill_ratio();
        current_state_.last_latency_ns = stats_window_.window_size() > 0 ? 
            static_cast<uint64_t>(stats_window_.get_mean()) : 0;
        current_state_.latency_variance = stats_window_.get_variance();
        current_state_.mean_latency_ns = static_cast<uint64_t>(stats_window_.get_mean());
        current_state_.p99_latency_ns = stats_window_.get_percentile(99.0);
        current_state_.p999_latency_ns = stats_window_.get_percentile(99.9);
    }
};

}
