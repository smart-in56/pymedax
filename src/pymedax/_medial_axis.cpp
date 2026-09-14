#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

namespace py = pybind11;

namespace {

using Index = std::ptrdiff_t;

struct QueueEntry {
    double distance;
    std::int64_t pixel;
    std::int64_t source;
};

struct QueueEntryGreater {
    bool operator()(const QueueEntry& left, const QueueEntry& right) const {
        if (left.distance != right.distance) {
            return left.distance > right.distance;
        }
        if (left.pixel != right.pixel) {
            return left.pixel > right.pixel;
        }
        return left.source > right.source;
    }
};

using Queue = std::priority_queue<
    QueueEntry, std::vector<QueueEntry>, QueueEntryGreater>;

constexpr int kNeighborCount = 8;
constexpr int kNeighborRows[kNeighborCount] = {-1, -1, -1, 0, 0, 1, 1, 1};
constexpr int kNeighborCols[kNeighborCount] = {-1, 0, 1, -1, 1, -1, 0, 1};

bool is_boundary(const std::uint8_t* image, Index rows, Index cols, Index row,
                 Index col) {
    if (image[row * cols + col] == 0) {
        return false;
    }

    for (int direction = 0; direction < kNeighborCount; ++direction) {
        const Index neighbor_row = row + kNeighborRows[direction];
        const Index neighbor_col = col + kNeighborCols[direction];
        if (neighbor_row < 0 || neighbor_row >= rows || neighbor_col < 0 ||
            neighbor_col >= cols ||
            image[neighbor_row * cols + neighbor_col] == 0) {
            return true;
        }
    }
    return false;
}

bool has_different_processed_front(
    const std::vector<std::uint8_t>& processed,
    const std::vector<std::int64_t>& closest_source, Index rows, Index cols,
    std::int64_t pixel, std::int64_t source) {
    const Index row = static_cast<Index>(pixel / cols);
    const Index col = static_cast<Index>(pixel % cols);

    for (int direction = 0; direction < kNeighborCount; ++direction) {
        const Index neighbor_row = row + kNeighborRows[direction];
        const Index neighbor_col = col + kNeighborCols[direction];
        if (neighbor_row < 0 || neighbor_row >= rows || neighbor_col < 0 ||
            neighbor_col >= cols) {
            continue;
        }
        const std::int64_t neighbor = neighbor_row * cols + neighbor_col;
        if (processed[neighbor] && closest_source[neighbor] != source) {
            return true;
        }
    }
    return false;
}

bool has_sufficient_front_angle(
    const std::vector<std::uint8_t>& processed,
    const std::vector<std::int64_t>& closest_source, Index rows, Index cols,
    std::int64_t pixel, double angle_threshold_degrees) {
    const Index row = static_cast<Index>(pixel / cols);
    const Index col = static_cast<Index>(pixel % cols);
    std::vector<std::int64_t> sources;

    const auto add_source = [&sources](std::int64_t source) {
        if (source >= 0 &&
            std::find(sources.begin(), sources.end(), source) == sources.end()) {
            sources.push_back(source);
        }
    };

    // The current pixel is processed, so its source is an arrived front too.
    add_source(closest_source[pixel]);

    // Ignore unprocessed neighbors: their source is not known yet.
    for (int direction = 0; direction < kNeighborCount; ++direction) {
        const Index neighbor_row = row + kNeighborRows[direction];
        const Index neighbor_col = col + kNeighborCols[direction];
        if (neighbor_row < 0 || neighbor_row >= rows || neighbor_col < 0 ||
            neighbor_col >= cols) {
            continue;
        }
        const std::int64_t neighbor = neighbor_row * cols + neighbor_col;
        if (processed[neighbor]) {
            add_source(closest_source[neighbor]);
        }
    }

    if (sources.size() < 2) {
        return false;
    }

    // Compute the direction of every arrived source relative to the current
    // pixel. The largest pairwise angle identifies the strongest front merge.
    std::vector<double> angles;
    angles.reserve(sources.size());
    for (const std::int64_t source : sources) {
        const Index source_row = static_cast<Index>(source / cols);
        const Index source_col = static_cast<Index>(source % cols);
        const double row_delta = static_cast<double>(source_row - row);
        const double col_delta = static_cast<double>(source_col - col);
        angles.push_back(std::atan2(row_delta, col_delta));
    }

    constexpr double pi = 3.14159265358979323846;
    double max_angle = 0.0;
    for (size_t i = 0; i < angles.size(); ++i) {
        for (size_t j = i + 1; j < angles.size(); ++j) {
            double angle = std::fabs(angles[i] - angles[j]);
            if (angle > pi) {
                angle = 2 * pi - angle;
            }
            if (angle > max_angle) {
                max_angle = angle;
            }
        }
    }

    const double angle_degrees = max_angle * 180.0 / pi;
    return angle_degrees >= angle_threshold_degrees;
}

py::tuple compute_medial_axis(
    py::array_t<std::uint8_t, py::array::c_style> image_array,
    double angle_threshold_degrees, double minimum_distance) {
    const py::buffer_info image_info = image_array.request();
    if (image_info.ndim != 2) {
        throw py::value_error("image must be a 2D NumPy array");
    }

    const Index rows = static_cast<Index>(image_info.shape[0]);
    const Index cols = static_cast<Index>(image_info.shape[1]);
    const std::size_t pixel_count = static_cast<std::size_t>(rows * cols);
    const auto* image = static_cast<const std::uint8_t*>(image_info.ptr);

    const double infinity = std::numeric_limits<double>::infinity();
    std::vector<double> distance(pixel_count, infinity);
    std::vector<std::int64_t> closest_source(pixel_count, -1);
    std::vector<std::uint8_t> processed(pixel_count, 0);
    std::vector<std::uint8_t> axis(pixel_count, 0);
    Queue queue;

    for (Index row = 0; row < rows; ++row) {
        for (Index col = 0; col < cols; ++col) {
            const std::int64_t pixel = row * cols + col;
            if (!is_boundary(image, rows, cols, row, col)) {
                continue;
            }
            distance[pixel] = 0.0;
            closest_source[pixel] = pixel;
            queue.push({0.0, pixel, pixel});
        }
    }

    while (!queue.empty()) {
        const QueueEntry current = queue.top();
        queue.pop();

        if (processed[current.pixel] || current.distance != distance[current.pixel] ||
            current.source != closest_source[current.pixel]) {
            continue;
        }

        processed[current.pixel] = 1;
        if (distance[current.pixel] >= minimum_distance &&
            has_different_processed_front(processed, closest_source, rows, cols,
                           current.pixel, current.source) &&
            has_sufficient_front_angle(processed, closest_source, rows, cols,
                           current.pixel,
                           angle_threshold_degrees)) {
            axis[current.pixel] = 1;
        }

        const Index row = static_cast<Index>(current.pixel / cols);
        const Index col = static_cast<Index>(current.pixel % cols);
        const Index source_row = static_cast<Index>(current.source / cols);
        const Index source_col = static_cast<Index>(current.source % cols);

        for (int direction = 0; direction < kNeighborCount; ++direction) {
            const Index neighbor_row = row + kNeighborRows[direction];
            const Index neighbor_col = col + kNeighborCols[direction];
            if (neighbor_row < 0 || neighbor_row >= rows || neighbor_col < 0 ||
                neighbor_col >= cols) {
                continue;
            }

            const std::int64_t neighbor = neighbor_row * cols + neighbor_col;
            if (image[neighbor] == 0 || processed[neighbor]) {
                continue;
            }

            const double row_delta = static_cast<double>(neighbor_row - source_row);
            const double col_delta = static_cast<double>(neighbor_col - source_col);
            const double candidate_distance =
                std::sqrt(row_delta * row_delta + col_delta * col_delta);
            if (candidate_distance < distance[neighbor]) {
                distance[neighbor] = candidate_distance;
                closest_source[neighbor] = current.source;
                queue.push({candidate_distance, neighbor, current.source});
            }
        }
    }

    py::array_t<std::uint8_t> axis_array({rows, cols});
    py::array_t<double> distance_array({rows, cols});
    py::array_t<std::int64_t> source_array({rows, cols});
    std::copy(axis.begin(), axis.end(), axis_array.mutable_data());
    std::copy(distance.begin(), distance.end(), distance_array.mutable_data());
    std::copy(closest_source.begin(), closest_source.end(), source_array.mutable_data());
    return py::make_tuple(std::move(axis_array), std::move(distance_array),
                          std::move(source_array));
}

}  // namespace

PYBIND11_MODULE(_medial_axis, module) {
    module.doc() = "C++ medial-axis propagation for binary 2D images";
    module.def(
          "medial_axis",
          [](py::array image, bool return_maps,
              double angle_threshold_degrees,
              double minimum_distance) -> py::object {
            if (image.ndim() != 2) {
                throw py::value_error("image must be a 2D NumPy array");
            }
            if (!image.dtype().is(py::dtype::of<std::uint8_t>())) {
                throw py::type_error("image must have dtype uint8");
            }
            if ((image.flags() & py::array::c_style) == 0) {
                throw py::value_error("image must be C-contiguous");
            }
            if (!std::isfinite(angle_threshold_degrees) ||
                angle_threshold_degrees < 0.0 ||
                angle_threshold_degrees > 180.0) {
                throw py::value_error(
                    "angle_threshold_degrees must be between 0 and 180");
            }
            if (!std::isfinite(minimum_distance) || minimum_distance < 0.0) {
                throw py::value_error(
                    "minimum_distance must be finite and non-negative");
            }

            py::array_t<std::uint8_t, py::array::c_style> typed_image(image);
            py::tuple result = compute_medial_axis(
                std::move(typed_image), angle_threshold_degrees,
                minimum_distance);
            if (return_maps) {
                return result;
            }
            return result[0].cast<py::object>();
        },
        py::arg("image"), py::arg("return_maps") = false,
        py::arg("angle_threshold_degrees") = 45.0,
        py::arg("minimum_distance") = 0.0,
        "Extract a medial axis from a contiguous uint8 2D image.");
}
