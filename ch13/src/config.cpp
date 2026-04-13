#include "myslam/config.h"

namespace myslam {
bool Config::SetParameterFile(const std::string &filename) {
    if (config_ == nullptr)
        config_ = std::shared_ptr<Config>(new Config);
    config_->file_ = cv::FileStorage(filename.c_str(), cv::FileStorage::READ);
    if (config_->file_.isOpened() == false) {
        LOG(ERROR) << "parameter file " << filename << " does not exist.";
        config_->file_.release();
        return false;
    }
    return true;
}

void Config::PrintAllParameters() {
    if (config_ == nullptr || !config_->file_.isOpened()) {
        LOG(INFO) << "[Config] No config file loaded.";
        return;
    }
    LOG(INFO) << "[Config] Loaded parameters:";
    cv::FileNode root = config_->file_.root();
    for (auto it = root.begin(); it != root.end(); ++it) {
        cv::FileNode node = *it;
        std::string key = node.name();
        if (node.isInt()) {
            LOG(INFO) << "  " << key << ": " << (int)node;
        } else if (node.isReal()) {
            LOG(INFO) << "  " << key << ": " << (double)node;
        } else if (node.isString()) {
            LOG(INFO) << "  " << key << ": " << (std::string)node;
        } else if (node.isSeq()) {
            LOG(INFO) << "  " << key << ": [array/sequence]";
        } else if (node.isMap()) {
            LOG(INFO) << "  " << key << ": [map/dict]";
        } else {
            LOG(INFO) << "  " << key << ": [unknown type]";
        }
    }
}

Config::~Config() {
    if (file_.isOpened())
        file_.release();
}

std::shared_ptr<Config> Config::config_ = nullptr;

}
