const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

// Allow .onnx files to be bundled
config.resolver.assetExts.push('onnx');

module.exports = config;