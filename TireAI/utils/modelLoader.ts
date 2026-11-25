import * as FileSystem from 'expo-file-system';
import * as ImageManipulator from 'expo-image-manipulator';

// Configure this address in TireAI/.env.local; see the root README.
const API_URL = process.env.EXPO_PUBLIC_API_URL?.replace(/\/+$/, '');

function getApiUrl(): string {
  if (!API_URL) {
    throw new Error('Set EXPO_PUBLIC_API_URL in TireAI/.env.local to the API address.');
  }
  return API_URL;
}

export interface PredictResult {
  label:      string;
  confidence: string;
  isCracked:  boolean;
  allScores:  { class: string; prob: number }[];
}

export async function loadModel(): Promise<boolean> {
  try {
    const res = await fetch(`${getApiUrl()}/`);
    const data = await res.json();
    console.log('✅ API connected:', data.status);
    return true;
  } catch (e) {
    throw new Error('Cannot connect to API. Check your IP and WiFi.');
  }
}

export async function predict(
  _model: any,
  imageUri: string
): Promise<PredictResult> {
  // Resize image before sending
  const resized = await ImageManipulator.manipulateAsync(
    imageUri,
    [{ resize: { width: 224, height: 224 } }],
    { format: ImageManipulator.SaveFormat.JPEG, compress: 0.8 }
  );

  // Send to API
  const formData = new FormData();
  formData.append('file', {
    uri:  resized.uri,
    name: 'tire.jpg',
    type: 'image/jpeg',
  } as any);

  const response = await fetch(`${getApiUrl()}/predict`, {
    method: 'POST',
    body:   formData,
  });

  if (!response.ok) {
    throw new Error(`Server error: ${response.status}`);
  }

  const result = await response.json();
  return {
    ...result,
    confidence: String(result.confidence),
  };
}