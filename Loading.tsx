import React from 'react';
import { View, Text, ActivityIndicator, StyleSheet } from 'react-native';

interface LoadingProps {
  isLoading: boolean;
  statusText: string;
}

const Loading: React.FC<LoadingProps> = ({ isLoading, statusText }) => {
  return (
    <View style={styles.loadingContainer}>
      {isLoading && (
        <>
          <ActivityIndicator size="large" color="#0000ff" />
          <Text>{statusText}</Text>
        </>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  loadingContainer: {
    position: 'absolute',
    left: 0,
    right: 0,
    top: 0,
    bottom: 0,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: 'rgba(255, 255, 255, 0.5)',
  },
});

export default Loading;
