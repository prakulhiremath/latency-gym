#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <latency_gym/engine.hpp>

namespace py = pybind11;
using namespace latency_gym;

PYBIND11_MODULE(_latency_gym, m) {
    m.doc() = "High-performance HFT latency simulator for Gymnasium environments";

    py::class_<TimeCounter>(m, "TimeCounter")
        .def(py::init<>())
        .def(py::init<uint64_t>())
        .def_readwrite("nanoseconds", &TimeCounter::nanoseconds)
        .def("__add__", &TimeCounter::operator+)
        .def("__sub__", &TimeCounter::operator-)
        .def("__lt__", &TimeCounter::operator<);

    py::class_<Order>(m, "Order")
        .def(py::init<>())
        .def(py::init<uint64_t, int64_t, uint32_t, bool>())
        .def_readwrite("order_id", &Order::order_id)
        .def_readwrite("timestamp_ns", &Order::timestamp_ns)
        .def_readwrite("size", &Order::size)
        .def_readwrite("is_buy", &Order::is_buy)
        .def_readwrite("match_time_ns", &Order::match_time_ns);

    py::class_<OrderRingBuffer>(m, "OrderRingBuffer")
        .def(py::init<size_t>())
        .def("push", &OrderRingBuffer::push)
        .def("pop", [](OrderRingBuffer& buf) -> py::object {
            Order order;
            if (buf.pop(order)) {
                return py::cast(order);
            }
            return py::none();
        })
        .def("size", &OrderRingBuffer::size)
        .def("capacity", &OrderRingBuffer::capacity)
        .def("overflow_count", &OrderRingBuffer::overflow_count)
        .def("is_full", &OrderRingBuffer::is_full)
        .def("is_empty", &OrderRingBuffer::is_empty)
        .def("fill_ratio", &OrderRingBuffer::fill_ratio);

    py::class_<LatencyStatsWindow>(m, "LatencyStatsWindow")
        .def(py::init<size_t>(), py::arg("window_size") = 1000)
        .def("record", &LatencyStatsWindow::record)
        .def("get_variance", &LatencyStatsWindow::get_variance)
        .def("get_mean", &LatencyStatsWindow::get_mean)
        .def("get_percentile", &LatencyStatsWindow::get_percentile)
        .def("window_size", &LatencyStatsWindow::window_size);

    py::class_<EnvironmentState>(m, "EnvironmentState")
        .def(py::init<>())
        .def_readwrite("queue_depth", &EnvironmentState::queue_depth)
        .def_readwrite("last_latency_ns", &EnvironmentState::last_latency_ns)
        .def_readwrite("latency_variance", &EnvironmentState::latency_variance)
        .def_readwrite("packet_drops", &EnvironmentState::packet_drops)
        .def_readwrite("queue_fill_ratio", &EnvironmentState::queue_fill_ratio)
        .def_readwrite("mean_latency_ns", &EnvironmentState::mean_latency_ns)
        .def_readwrite("p99_latency_ns", &EnvironmentState::p99_latency_ns)
        .def_readwrite("p999_latency_ns", &EnvironmentState::p999_latency_ns);

    py::class_<LatencySimulator>(m, "LatencySimulator")
        .def(py::init<uint32_t>(), py::arg("seed") = 42)
        .def("reset", &LatencySimulator::reset)
        .def("set_action", &LatencySimulator::set_action)
        .def("step", &LatencySimulator::step)
        .def("get_state", &LatencySimulator::get_state, py::return_value_policy::reference_internal)
        .def("compute_reward", &LatencySimulator::compute_reward, py::arg("alpha") = 1.0, py::arg("beta") = 0.5, py::arg("gamma") = 2.0)
        .def("get_step_count", &LatencySimulator::get_step_count)
        .def("get_last_reward", &LatencySimulator::get_last_reward)
        .def("get_matched_orders", &LatencySimulator::get_matched_orders)
        .def("get_total_overflow", &LatencySimulator::get_total_overflow)
        .def_readonly_static("DEFAULT_BUFFER_CAPACITY", &LatencySimulator::DEFAULT_BUFFER_CAPACITY)
        .def_readonly_static("TIME_STEP_NS", &LatencySimulator::TIME_STEP_NS)
        .def_readonly_static("MIN_BATCH_SIZE", &LatencySimulator::MIN_BATCH_SIZE)
        .def_readonly_static("MAX_BATCH_SIZE", &LatencySimulator::MAX_BATCH_SIZE)
        .def_readonly_static("MIN_POLLING_RATE", &LatencySimulator::MIN_POLLING_RATE)
        .def_readonly_static("MAX_POLLING_RATE", &LatencySimulator::MAX_POLLING_RATE)
        .def_readonly_static("MIN_PREALLOC_POOL", &LatencySimulator::MIN_PREALLOC_POOL)
        .def_readonly_static("MAX_PREALLOC_POOL", &LatencySimulator::MAX_PREALLOC_POOL);
}
