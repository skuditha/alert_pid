#include "DataWriter.h"

#include <cstring>
#include <string>
#include <vector>

#include <hdf5.h>

namespace alert::postpid {

class DataWriter::Impl {
public:
    hid_t file = -1;

    hid_t d_features = -1;
    hid_t d_masks = -1;
    hid_t d_class_index = -1;
    hid_t d_truth_pid = -1;
    hid_t d_event_index = -1;
    hid_t d_track_id = -1;
    hid_t d_hit_id = -1;
    hid_t d_cluster_id = -1;
    hid_t d_status = -1;
    hid_t d_has_masked = -1;

    hsize_t n_rows = 0;
};

namespace {

bool writeStringAttribute(hid_t obj, const char* name, const std::string& value) {
    hid_t type = H5Tcopy(H5T_C_S1);
    H5Tset_size(type, value.size());
    hid_t space = H5Screate(H5S_SCALAR);
    hid_t attr = H5Acreate2(obj, name, type, space, H5P_DEFAULT, H5P_DEFAULT);
    if (attr < 0) {
        H5Sclose(space);
        H5Tclose(type);
        return false;
    }
    const herr_t status = H5Awrite(attr, type, value.c_str());
    H5Aclose(attr);
    H5Sclose(space);
    H5Tclose(type);
    return status >= 0;
}

hid_t create1D(hid_t parent, const char* name, hid_t type, hsize_t n) {
    const hsize_t dims[1] = {n};
    hid_t space = H5Screate_simple(1, dims, nullptr);
    hid_t ds = H5Dcreate2(parent, name, type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(space);
    return ds;
}

hid_t create2D(hid_t parent, const char* name, hid_t type, hsize_t n0, hsize_t n1) {
    const hsize_t dims[2] = {n0, n1};
    hid_t space = H5Screate_simple(2, dims, nullptr);
    hid_t ds = H5Dcreate2(parent, name, type, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(space);
    return ds;
}

template <typename T>
bool writeRow1D(hid_t dataset, hsize_t row, hid_t mem_type, const T* value) {
    hid_t filespace = H5Dget_space(dataset);
    const hsize_t start[1] = {row};
    const hsize_t count[1] = {1};
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, nullptr, count, nullptr);

    hid_t memspace = H5Screate_simple(1, count, nullptr);
    const herr_t status = H5Dwrite(dataset, mem_type, memspace, filespace, H5P_DEFAULT, value);

    H5Sclose(memspace);
    H5Sclose(filespace);
    return status >= 0;
}

template <typename T>
bool writeRow2D(hid_t dataset, hsize_t row, hsize_t ncols, hid_t mem_type, const T* values) {
    hid_t filespace = H5Dget_space(dataset);
    const hsize_t start[2] = {row, 0};
    const hsize_t count[2] = {1, ncols};
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, start, nullptr, count, nullptr);

    hid_t memspace = H5Screate_simple(2, count, nullptr);
    const herr_t status = H5Dwrite(dataset, mem_type, memspace, filespace, H5P_DEFAULT, values);

    H5Sclose(memspace);
    H5Sclose(filespace);
    return status >= 0;
}

}  // namespace

DataWriter::DataWriter() : impl_(new Impl()) {}
DataWriter::~DataWriter() {
    close();
    delete impl_;
    impl_ = nullptr;
}

bool DataWriter::open(const std::string& output_path, std::size_t n_rows) {
    impl_->n_rows = static_cast<hsize_t>(n_rows);
    impl_->file = H5Fcreate(output_path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (impl_->file < 0) {
        return false;
    }

    hid_t g_features = H5Gcreate2(impl_->file, "/features", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t g_labels   = H5Gcreate2(impl_->file, "/labels", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t g_rowmeta  = H5Gcreate2(impl_->file, "/row_meta", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t g_cutflow  = H5Gcreate2(impl_->file, "/cutflow", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t g_meta     = H5Gcreate2(impl_->file, "/dataset_meta", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    if (g_features < 0 || g_labels < 0 || g_rowmeta < 0 || g_cutflow < 0 || g_meta < 0) {
        return false;
    }

    impl_->d_features = create2D(g_features, "values", H5T_IEEE_F32LE, impl_->n_rows, kNumFeatures);
    impl_->d_masks = create2D(g_features, "masks", H5T_STD_U8LE, impl_->n_rows, kNumFeatures);

    impl_->d_class_index = create1D(g_labels, "class_index", H5T_STD_I32LE, impl_->n_rows);
    impl_->d_truth_pid = create1D(g_labels, "truth_pid", H5T_STD_I32LE, impl_->n_rows);

    impl_->d_event_index = create1D(g_rowmeta, "event_index", H5T_STD_I64LE, impl_->n_rows);
    impl_->d_track_id = create1D(g_rowmeta, "track_id", H5T_STD_I32LE, impl_->n_rows);
    impl_->d_hit_id = create1D(g_rowmeta, "matched_atof_hit_id", H5T_STD_I32LE, impl_->n_rows);
    impl_->d_cluster_id = create1D(g_rowmeta, "cluster_id", H5T_STD_I32LE, impl_->n_rows);
    impl_->d_status = create1D(g_rowmeta, "status", H5T_STD_I32LE, impl_->n_rows);
    impl_->d_has_masked = create1D(g_rowmeta, "has_any_masked_feature", H5T_STD_U8LE, impl_->n_rows);

    // Write feature names as attributes on /features.
    std::string joined_names;
    for (int i = 0; i < kNumFeatures; ++i) {
        if (i) joined_names += ",";
        joined_names += kFeatureNames[i];
    }
    writeStringAttribute(g_features, "feature_names_csv", joined_names);

    H5Gclose(g_features);
    H5Gclose(g_labels);
    H5Gclose(g_rowmeta);
    H5Gclose(g_cutflow);
    H5Gclose(g_meta);

    return true;
}

bool DataWriter::writeRow(std::size_t index, const FeatureRow& features, const OutputRowMeta& meta) {
    const hsize_t row = static_cast<hsize_t>(index);

    bool ok = true;
    ok &= writeRow2D(impl_->d_features, row, kNumFeatures, H5T_NATIVE_FLOAT, features.values.data());
    ok &= writeRow2D(impl_->d_masks, row, kNumFeatures, H5T_NATIVE_UINT8, features.masks.data());

    ok &= writeRow1D(impl_->d_class_index, row, H5T_NATIVE_INT32, &meta.class_index);
    ok &= writeRow1D(impl_->d_truth_pid, row, H5T_NATIVE_INT32, &meta.truth_pid);

    ok &= writeRow1D(impl_->d_event_index, row, H5T_NATIVE_INT64, &meta.event_index);
    ok &= writeRow1D(impl_->d_track_id, row, H5T_NATIVE_INT32, &meta.track_id);
    ok &= writeRow1D(impl_->d_hit_id, row, H5T_NATIVE_INT32, &meta.matched_atof_hit_id);
    ok &= writeRow1D(impl_->d_cluster_id, row, H5T_NATIVE_INT32, &meta.cluster_id);
    ok &= writeRow1D(impl_->d_status, row, H5T_NATIVE_INT32, &meta.status);
    ok &= writeRow1D(impl_->d_has_masked, row, H5T_NATIVE_UINT8, &meta.has_any_masked_feature);

    return ok;
}

bool DataWriter::writeMetadata(
    const std::vector<std::string>& input_files,
    const Cutflow& cutflow,
    const std::string& label_map_version,
    const std::string& feature_contract_version)
{
    hid_t g_cutflow = H5Gopen2(impl_->file, "/cutflow", H5P_DEFAULT);
    hid_t g_meta = H5Gopen2(impl_->file, "/dataset_meta", H5P_DEFAULT);
    if (g_cutflow < 0 || g_meta < 0) {
        return false;
    }

    writeStringAttribute(g_meta, "label_map_version", label_map_version);
    writeStringAttribute(g_meta, "feature_contract_version", feature_contract_version);

    std::string joined_inputs;
    for (std::size_t i = 0; i < input_files.size(); ++i) {
        if (i) joined_inputs += ",";
        joined_inputs += input_files[i];
    }
    writeStringAttribute(g_meta, "input_files_csv", joined_inputs);

    const int32_t n_features = kNumFeatures;
    hid_t ds_nf = create1D(g_meta, "n_features", H5T_STD_I32LE, 1);
    H5Dwrite(ds_nf, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, &n_features);
    H5Dclose(ds_nf);

    for (const auto& [key, value] : cutflow.counters()) {
        hid_t ds = create1D(g_cutflow, key.c_str(), H5T_STD_I64LE, 1);
        H5Dwrite(ds, H5T_NATIVE_INT64, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);
        H5Dclose(ds);
    }

    H5Gclose(g_cutflow);
    H5Gclose(g_meta);
    return true;
}

bool DataWriter::close() {
    if (!impl_) {
        return true;
    }

    auto close_ds = [](hid_t& x) {
        if (x >= 0) {
            H5Dclose(x);
            x = -1;
        }
    };

    close_ds(impl_->d_features);
    close_ds(impl_->d_masks);
    close_ds(impl_->d_class_index);
    close_ds(impl_->d_truth_pid);
    close_ds(impl_->d_event_index);
    close_ds(impl_->d_track_id);
    close_ds(impl_->d_hit_id);
    close_ds(impl_->d_cluster_id);
    close_ds(impl_->d_status);
    close_ds(impl_->d_has_masked);

    if (impl_->file >= 0) {
        H5Fclose(impl_->file);
        impl_->file = -1;
    }

    return true;
}

}  // namespace alert::postpid