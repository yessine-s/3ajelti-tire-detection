import React, { useState, useEffect } from 'react';
import {
  View, Text, Image, TouchableOpacity,
  StyleSheet, ActivityIndicator, ScrollView, Alert
} from 'react-native';
import * as ImagePicker from 'expo-image-picker';
import { loadModel, predict, PredictResult } from '../../utils/modelLoader';

export default function HomeScreen() {
  const [ready,    setReady]    = useState(false);
  const [imageUri, setImageUri] = useState<string | null>(null);
  const [result,   setResult]   = useState<PredictResult | null>(null);
  const [loading,  setLoading]  = useState(false);

  useEffect(() => {
    (async () => {
      try {
        await loadModel();
        setReady(true);
      } catch (e: any) {
        Alert.alert('Connection Error', e.message);
      }
    })();
  }, []);

  const pickImage = async (fromCamera: boolean) => {
    const perm = fromCamera
      ? await ImagePicker.requestCameraPermissionsAsync()
      : await ImagePicker.requestMediaLibraryPermissionsAsync();

    if (!perm.granted) {
      Alert.alert('Permission needed');
      return;
    }

    const picked = fromCamera
      ? await ImagePicker.launchCameraAsync({ quality: 0.8 })
      : await ImagePicker.launchImageLibraryAsync({ quality: 0.8 });

    if (!picked.canceled) {
      const uri = picked.assets[0].uri;
      setImageUri(uri);
      setResult(null);
      setLoading(true);
      try {
        const res = await predict(null, uri);
        setResult(res);
      } catch (e: any) {
        Alert.alert('Error', e.message);
      }
      setLoading(false);
    }
  };

  const getColor  = () => !result ? '#F8F9FA'
    : result.label === 'Non-defective' ? '#F0FFF4'
    : result.label === 'Cracks'        ? '#FFF8E1' : '#FFF0F0';

  const getIcon   = () => !result ? ''
    : result.label === 'Non-defective' ? '✅'
    : result.label === 'Cracks'        ? '⚠️' : '🔴';

  const getProblemInfo = () => {
    if (!result) return { problem: '', danger: '', nextStep: '', color: '' };

    if (result.label === 'Non-defective') {
      return {
        problem: 'No visible defect was detected in this image. This does not confirm that the tire is safe.',
        danger: 'No defect detected',
        nextStep: 'Continue regular inspections',
        color: '#27AE60'
      };
    } else if (result.label === 'Cracks') {
      return {
        problem: 'Possible tire damage detected. Have the tire inspected by a qualified tire professional.',
        danger: 'Inspection needed',
        nextStep: 'Arrange inspection promptly',
        color: '#F39C12'
      };
    } else {
      return {
        problem: 'Possible sidewall bulge detected. Avoid using this tire and have it inspected by a qualified tire professional.',
        danger: 'Urgent inspection',
        nextStep: 'Have it inspected before reuse',
        color: '#E74C3C'
      };
    }
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.title}>🛞 3ajeltii</Text>
      <Text style={styles.subtitle}>
        {ready ? '✅ API Connected' : '⏳ Connecting to server...'}
      </Text>

      {imageUri
        ? <Image source={{ uri: imageUri }} style={styles.image} />
        : <View style={styles.placeholder}>
            <Text style={styles.placeholderIcon}>📷</Text>
            <Text style={styles.placeholderText}>No image selected</Text>
          </View>
      }

      {loading && (
        <View style={styles.loadingBox}>
          <ActivityIndicator size="large" color="#2C3E50" />
          <Text style={styles.loadingText}>Analyzing tire...</Text>
        </View>
      )}

      {!loading && result && (
        <View style={[styles.card, { backgroundColor: getColor() }]}>
          <Text style={styles.cardIcon}>{getIcon()}</Text>
          <Text style={styles.cardLabel}>{result.label}</Text>
          <Text style={styles.cardConf}>Confidence: {result.confidence}%</Text>

          <View style={styles.problemBox}>
            <Text style={styles.problemTitle}>Problem:</Text>
            <Text style={styles.problemText}>{getProblemInfo().problem}</Text>

            <View style={styles.infoRow}>
              <View style={styles.infoPart}>
                <Text style={styles.infoLabel}>Model signal</Text>
                <Text style={[styles.infoValue, { color: getProblemInfo().color }]}>
                  {getProblemInfo().danger}
                </Text>
              </View>
              <View style={styles.infoPart}>
                <Text style={styles.infoLabel}>Next step</Text>
                <Text style={[styles.infoValue, { color: getProblemInfo().color }]}>
                  {getProblemInfo().nextStep}
                </Text>
              </View>
            </View>
          </View>

          <View style={styles.scoresBox}>
            {result.allScores.map(s => (
              <View key={s.class} style={styles.scoreRow}>
                <Text style={styles.scoreLabel}>{s.class}</Text>
                <View style={styles.barBg}>
                  <View style={[styles.barFill, { width: `${s.prob}%` as any }]} />
                </View>
                <Text style={styles.scoreVal}>{s.prob}%</Text>
              </View>
            ))}
          </View>
        </View>
      )}

      <TouchableOpacity
        style={[styles.btn, !ready && styles.btnOff]}
        onPress={() => pickImage(true)}
        disabled={!ready || loading}
      >
        <Text style={styles.btnText}>📷 Take Photo</Text>
      </TouchableOpacity>

      <TouchableOpacity
        style={[styles.btn, styles.btnGreen, !ready && styles.btnOff]}
        onPress={() => pickImage(false)}
        disabled={!ready || loading}
      >
        <Text style={styles.btnText}>🖼️ Pick from Gallery</Text>
      </TouchableOpacity>

      {imageUri && !loading && (
        <TouchableOpacity
          style={[styles.btn, styles.btnOrange]}
          onPress={() => { setImageUri(null); setResult(null); }}
        >
          <Text style={styles.btnText}>🔄 Reset</Text>
        </TouchableOpacity>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container:       { flexGrow:1, alignItems:'center', padding:24, backgroundColor:'#F8F9FA' },
  title:           { fontSize:26, fontWeight:'bold', color:'#2C3E50', marginTop:20, marginBottom:4 },
  subtitle:        { fontSize:14, color:'#7F8C8D', marginBottom:20 },
  image:           { width:300, height:300, borderRadius:16, marginBottom:12 },
  placeholder:     { width:300, height:300, borderRadius:16, backgroundColor:'#ECF0F1',
                     alignItems:'center', justifyContent:'center', marginBottom:12 },
  placeholderIcon: { fontSize:48 },
  placeholderText: { color:'#95A5A6', fontSize:16, marginTop:8 },
  loadingBox:      { alignItems:'center', marginVertical:12 },
  loadingText:     { marginTop:8, color:'#555' },
  card:            { width:'100%', borderRadius:16, padding:20, alignItems:'center',
                     marginBottom:16, elevation:4 },
  cardIcon:        { fontSize:48, marginBottom:6 },
  cardLabel:       { fontSize:24, fontWeight:'bold', color:'#2C3E50', marginBottom:4 },
  cardConf:        { fontSize:15, color:'#555', marginBottom:6 },
  problemBox:      { width:'100%', backgroundColor:'rgba(255,255,255,0.6)', borderRadius:12, padding:12, marginBottom:16 },
  problemTitle:    { fontSize:14, fontWeight:'bold', color:'#2C3E50', marginBottom:6 },
  problemText:     { fontSize:13, color:'#444', lineHeight:20, marginBottom:12 },
  infoRow:         { flexDirection:'row', gap:12, marginTop:8 },
  infoPart:        { flex:1, alignItems:'center', paddingVertical:8, borderRadius:8, backgroundColor:'rgba(255,255,255,0.5)' },
  infoLabel:       { fontSize:11, color:'#666', marginBottom:4, fontWeight:'600' },
  infoValue:       { fontSize:12, fontWeight:'bold' },
  scoresBox:       { width:'100%', gap:8 },
  scoreRow:        { flexDirection:'row', alignItems:'center', gap:8 },
  scoreLabel:      { width:110, fontSize:13, color:'#444' },
  barBg:           { flex:1, height:10, backgroundColor:'#DDD', borderRadius:5, overflow:'hidden' },
  barFill:         { height:10, backgroundColor:'#2C3E50', borderRadius:5 },
  scoreVal:        { width:45, fontSize:13, color:'#444', textAlign:'right' },
  btn:             { backgroundColor:'#2C3E50', padding:16, borderRadius:12,
                     width:'100%', alignItems:'center', marginTop:10 },
  btnGreen:        { backgroundColor:'#27AE60' },
  btnOrange:       { backgroundColor:'#E67E22' },
  btnOff:          { backgroundColor:'#BDC3C7' },
  btnText:         { color:'#fff', fontSize:16, fontWeight:'600' },
});