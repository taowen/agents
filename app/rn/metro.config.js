const path = require("path");
const { getDefaultConfig, mergeConfig } = require("@react-native/metro-config");

const repoRoot = path.resolve(__dirname, "../..");
const localModules = path.resolve(__dirname, "node_modules");

const config = {
  watchFolders: [repoRoot],
  resolver: {
    // Local node_modules first so react 19.2.3 wins over root's 19.2.4
    nodeModulesPaths: [localModules, path.resolve(repoRoot, "node_modules")],
    // Pin react to local version to match react-native-renderer
    extraNodeModules: {
      react: path.resolve(localModules, "react")
    }
  }
};

module.exports = mergeConfig(getDefaultConfig(__dirname), config);
