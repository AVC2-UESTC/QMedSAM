#include <iostream>
#include <filesystem>
#include <string>
#include <algorithm>
#include <vector>
#include <chrono>
#include <argparse/argparse.hpp>
#include <xtensor-io/xnpz.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xoperation.hpp>
#include <xtensor/xbuilder.hpp>
#include <xtensor/xmanipulation.hpp>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

int main(int argc, char *argv[]) {
    argparse::ArgumentParser program("uestcsd_inference");
    program.add_argument("-i", "--input_dir");
    program.add_argument("-o", "--output_dir");
    program.add_argument("-m", "--model_dir").default_value(".");
    program.add_argument("-b", "--block_size").default_value(96);
    try {
        program.parse_args(argc, argv);
    } catch (const std::exception& err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }
    int block_size = program.get<int>("-b");
    std::filesystem::path i_dir(program.get<std::string>("-i"));
    std::filesystem::path o_dir(program.get<std::string>("-o"));
    std::filesystem::path m_dir(program.get<std::string>("-m"));
    auto enc_fp = m_dir / "encoder.xml", dec_fp = m_dir / "decoder.xml";
    ov::Core core;
    ov::CompiledModel encoder = core.compile_model(enc_fp.string(), "CPU");
    ov::CompiledModel decoder = core.compile_model(dec_fp.string(), "CPU");
    ov::InferRequest encoder_infer_request = encoder.create_infer_request();
    ov::InferRequest decoder_infer_request = decoder.create_infer_request();
    if (!std::filesystem::exists(o_dir) && !std::filesystem::create_directory(o_dir)) {
        throw std::runtime_error("Fail to create " + o_dir.string());
    }
    for (const auto& entry : std::filesystem::directory_iterator(i_dir)) {
        auto infer_s = std::chrono::high_resolution_clock::now();
        if (!entry.is_regular_file() || entry.path().extension().string().compare(".npz")) continue;
        auto clip_path = entry.path();
        auto clip_save_path = o_dir / entry.path().filename();
        if (entry.path().filename().string()[0] == '2') {
            // continue;
            auto _npz = xt::load_npz(clip_path.string());
            xt::xarray<uint8_t> img_3c = _npz["imgs"].cast<uint8_t>();
            xt::xarray<int64_t> boxes = _npz["boxes"].cast<int64_t>();
            unsigned long oh = img_3c.shape()[0], ow = img_3c.shape()[1];
            unsigned long img_size = 256;
            float scale = 1.0f * img_size / std::max(oh, ow);
            unsigned long nh = scale * oh + 0.5, nw = scale * ow + 0.5;
            unsigned long ph = img_size - nh, pw = img_size - nw;
            unsigned long batches = boxes.shape()[0];
            xt::xarray<uint8_t> segs = xt::zeros<uint8_t>({oh, ow});
            xt::xarray<int64_t> boxes_points = boxes.reshape({batches, 2, 2});
            xt::xarray<float> tune_factor = xt::xarray<float>({{{1.0f * nw / ow, 1.0f * nh / oh}}});
            xt::xarray<float> boxes_resize = 1.0 * boxes_points * tune_factor;
            xt::xarray<float> bbox = boxes_resize.reshape({batches, 1, 4});
            cv::Mat mat1(cv::Size(ow, oh), CV_8UC3, img_3c.data()), mat2;
            cv::resize(mat1, mat2, cv::Size(nw, nh), cv::INTER_CUBIC);
            xt::xarray<float> img_resize = xt::adapt(
                (uint8_t*)mat2.data,
                mat2.total() * mat2.channels(),
                xt::no_ownership(),
                std::vector<int> {mat2.rows, mat2.cols, mat2.channels()}
            );
            xt::xarray<float> img_pad = xt::pad(img_resize, {{0, ph}, {0, pw}, {0, 0}});
            float imin = xt::amin(img_pad)(), imax = xt::amax(img_pad)();
            xt::xarray<float> img_norm = (img_pad - imin) / std::clamp(imax - imin, 1e-8f, 1e18f);
            xt::xarray<float> img_c3 = xt::expand_dims(xt::transpose(img_norm, {2, 0, 1}), 0);
            ov::Tensor ov_image(ov::element::f32, {1, 3, 256, 256}, img_c3.data());
            encoder_infer_request.set_tensor("image", ov_image);
            encoder_infer_request.infer();
            ov::Tensor image_embedding = encoder_infer_request.get_output_tensor();
            xt::xarray<float> low_res_masks = xt::empty<float>({0, 256, 256});
            int num_blocks = batches / block_size;
            if (batches % block_size) num_blocks++;
            for (int i = 0; i < num_blocks; i++) {
                unsigned long slice_s = i * block_size;
                unsigned long slice_t = i == num_blocks - 1 ? batches : (i + 1) * block_size;
                xt::xarray<float> bbox_i = xt::view(bbox, xt::range(slice_s, slice_t), xt::all(), xt::all());
                ov::Tensor ov_boxes_i(ov::element::f32, {slice_t - slice_s, 1, 4}, bbox_i.data());
                decoder_infer_request.set_tensor("image_embedding", image_embedding);
                decoder_infer_request.set_tensor("boxes", ov_boxes_i);
                decoder_infer_request.infer();
                xt::xarray<float> low_res_masks_i = xt::adapt(
                    decoder_infer_request.get_output_tensor().data<float>(),
                    (slice_t - slice_s) * 256 * 256,
                    xt::no_ownership(),
                    std::vector<unsigned long> {slice_t - slice_s, 256, 256}
                );
                low_res_masks = xt::concatenate(xt::xtuple(low_res_masks, low_res_masks_i), 0);
            }
            for (unsigned long i = 0; i < batches; i++) {
                xt::xarray<float> maski = xt::view(low_res_masks, i, xt::range(_, nh), xt::range(_, nw));
                cv::Mat mat3(cv::Size(nw, nh), CV_32FC1, maski.data()), mat4;
                cv::resize(mat3, mat4, cv::Size(ow, oh), cv::INTER_CUBIC);
                xt::xarray<float> pd = xt::adapt(
                    (float*)mat4.data,
                    mat4.total(),
                    xt::no_ownership(),
                    std::vector<int> {mat4.rows, mat4.cols}
                );
                xt::filtration(segs, pd > 0) = i + 1;
            }
            xt::dump_npz(clip_save_path.string(), "segs", segs, true);
        } else {
            auto _npz = xt::load_npz(clip_path.string());
            xt::xarray<uint8_t> img_3d = _npz["imgs"].cast<uint8_t>();
            xt::xarray<int64_t> boxes = _npz["boxes"].cast<int64_t>();
            unsigned long len_z = img_3d.shape()[0], oh = img_3d.shape()[1], ow = img_3d.shape()[2];
            unsigned long img_size = 256;
            float scale = 1.0f * img_size / std::max(oh, ow);
            unsigned long nh = scale * oh + 0.5, nw = scale * ow + 0.5;
            unsigned long ph = img_size - nh, pw = img_size - nw;
            unsigned long batches = boxes.shape()[0];
            xt::xarray<uint8_t> segs = xt::zeros<uint8_t>({len_z, oh, ow});
            auto fakenull = new xt::xarray<float>();
            std::vector<xt::xarray<float>*> box_z(len_z, fakenull);        
            std::vector<std::vector<int>> idx_z(len_z);
            for (int i = 0; i < batches; i++) {
                xt::xarray<int64_t> box = xt::view(boxes, i, xt::all());
                int64_t x_min = box(0), y_min = box(1), z_min = box(2), x_max = box(3), y_max = box(4), z_max = box(5);
                xt::xarray<float> box_resize = xt::xarray<float>({{{
                    1.0f * x_min * nw / ow, 1.0f * y_min * nh / oh,
                    1.0f * x_max * nw / ow, 1.0f * y_max * nh / oh
                }}});
                if (z_min < 0) z_min = 0;
                if (z_max >= len_z) z_max = len_z - 1;
                int64_t z_middle = (z_max - z_min) / 2 + z_min;
                int64_t z = z_middle;
                while (true) {
                    if (box_z[z] == fakenull) {
                        box_z[z] = new xt::xarray<float>(box_resize);
                    } else {
                        xt::xarray<float>* box_origin = box_z[z];
                        xt::xarray<float> stacked_box = xt::concatenate(xt::xtuple(*box_origin, box_resize), 0);
                        box_z[z] = new xt::xarray<float>(stacked_box);
                        delete box_origin;
                    }
                    idx_z[z].push_back(i + 1);
                    if (z == z_min) {
                        break;
                    } else if (z == z_max) {
                        z = z_middle - 1;
                    } else if (z >= z_middle) {
                        z += 1;
                    } else {
                        z -= 1;
                    }
                }
            }
            for (size_t z = 0; z < len_z; z++) {
                if (box_z[z] == fakenull) continue;
                xt::xarray<float>& bbox = *box_z[z];
                unsigned long batch_z = bbox.shape()[0];
                xt::xarray<uint8_t> img = xt::view(img_3d, z, xt::all(), xt::all());
                cv::Mat mat1(cv::Size(ow, oh), CV_8UC1, img.data()), mat2;
                cv::resize(mat1, mat2, cv::Size(nw, nh), cv::INTER_CUBIC);
                xt::xarray<float> img_resize = xt::adapt(
                    (uint8_t*)mat2.data,
                    mat2.total(),
                    xt::no_ownership(),
                    std::vector<int> {mat2.rows, mat2.cols}
                );
                xt::xarray<float> img_pad = xt::pad(img_resize, {{0, ph}, {0, pw}});
                float imin = xt::amin(img_pad)(), imax = xt::amax(img_pad)();
                xt::xarray<float> img_norm = (img_pad - imin) / std::clamp(imax - imin, 1e-8f, 1e18f);
                xt::xarray<float> img_c3 = xt::repeat(xt::expand_dims(xt::expand_dims(img_norm, 0), 0), 3, 1);
                ov::Tensor ov_image(ov::element::f32, {1, 3, 256, 256}, img_c3.data());
                encoder_infer_request.set_tensor("image", ov_image);
                encoder_infer_request.infer();
                ov::Tensor image_embedding = encoder_infer_request.get_output_tensor();
                xt::xarray<float> low_res_masks = xt::empty<float>({0, 256, 256});
                int num_blocks = batch_z / block_size;
                if (batch_z % block_size) num_blocks++;
                for (int i = 0; i < num_blocks; i++) {
                    unsigned long slice_s = i * block_size;
                    unsigned long slice_t = i == num_blocks - 1 ? batch_z : (i + 1) * block_size;
                    xt::xarray<float> bbox_i = xt::view(bbox, xt::range(slice_s, slice_t), xt::all(), xt::all());
                    ov::Tensor ov_boxes_i(ov::element::f32, {slice_t - slice_s, 1, 4}, bbox_i.data());
                    decoder_infer_request.set_tensor("image_embedding", image_embedding);
                    decoder_infer_request.set_tensor("boxes", ov_boxes_i);
                    decoder_infer_request.infer();
                    xt::xarray<float> low_res_masks_i = xt::adapt(
                        decoder_infer_request.get_output_tensor().data<float>(),
                        (slice_t - slice_s) * 256 * 256,
                        xt::no_ownership(),
                        std::vector<unsigned long> {slice_t - slice_s, 256, 256}
                    );
                    low_res_masks = xt::concatenate(xt::xtuple(low_res_masks, low_res_masks_i), 0);
                }
                for (unsigned long i = 0; i < batch_z; i++) {
                    xt::xarray<float> maski = xt::view(low_res_masks, i, xt::range(_, nh), xt::range(_, nw));
                    cv::Mat mat3(cv::Size(nw, nh), CV_32FC1, maski.data()), mat4;
                    cv::resize(mat3, mat4, cv::Size(ow, oh), cv::INTER_CUBIC);
                    xt::xarray<float> pd = xt::adapt(
                        (float*)mat4.data,
                        mat4.total(),
                        xt::no_ownership(),
                        std::vector<int> {mat4.rows, mat4.cols}
                    );
                    auto segs_z = xt::view(segs, z, xt::all(), xt::all());
                    xt::filtration(segs_z, pd > 0) = idx_z[z][i];
                }
                delete box_z[z];
            }
            delete fakenull;
            xt::dump_npz(clip_save_path.string(), "segs", segs, true);
        }
        auto infer_t = std::chrono::high_resolution_clock::now();
        std::cout << "Inference " << entry.path().filename().string() << " in ";
        std::cout << std::chrono::duration_cast<std::chrono::milliseconds> (infer_t - infer_s).count() << " ms\n";
    }
    return 0;
}
