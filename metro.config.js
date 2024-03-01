// Learn more https://docs.expo.io/guides/customizing-metro
const {getDefaultConfig} = require('expo/metro-config')

/** @type {import('expo/metro-config').MetroConfig} */

module.exports = (() => {
    const defaultConfig = getDefaultConfig(__dirname)
    const {assetExts, sourceExts} = defaultConfig.resolver
    return {
        resolver: {
            assetExts: [...assetExts, 'bin', 'css'],
            sourceExts: [...sourceExts, 'mjs', 'cjs']
        }
    }
})()