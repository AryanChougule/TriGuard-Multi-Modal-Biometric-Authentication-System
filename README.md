# TriGuard: Multi-Modal Biometric Authentication System

A secure AI-powered authentication system that combines **Face Recognition**, **Voice Verification**, and **Lip-Sync Validation** to provide robust multi-factor biometric authentication.

## Overview

Traditional authentication methods such as passwords and PINs are vulnerable to theft, phishing, and replay attacks. TriGuard addresses these challenges by combining three independent biometric modalities:

* Face Recognition
* Voice Recognition
* Lip-Sync Verification

Authentication is granted only when all required modalities pass verification, significantly reducing the risk of spoofing and unauthorized access.

---

## Features

### Face Authentication

* Face detection and alignment
* Face embedding extraction
* Identity verification using stored embeddings
* Real-time webcam support

### Voice Authentication

* Speaker verification using ECAPA-TDNN
* Voice embedding generation
* Enrollment and authentication workflows
* Robust speaker matching

### Lip-Sync Authentication

* Random challenge-response verification
* Real-time lip movement tracking
* Audio-visual synchronization checks
* Protection against replay attacks

### Multi-Modal Decision Engine

* Combines results from all authentication modules
* Configurable confidence thresholds
* Detailed authentication logs
* Real-time authentication feedback

---

## System Architecture

```text
User
 │
 ├── Face Module
 │      └── Face Embeddings
 │
 ├── Voice Module
 │      └── Speaker Embeddings
 │
 ├── Lip-Sync Module
 │      └── Audio-Visual Consistency Check
 │
 └── Decision Engine
          │
          ├── Access Granted
          └── Access Denied
```

---

## Authentication Workflow

### User Enrollment

1. Capture face images.
2. Record voice samples.
3. Generate facial embeddings.
4. Generate speaker embeddings.
5. Store encrypted biometric templates.

### User Authentication

1. User starts authentication.
2. Face verification is performed.
3. Voice verification is performed.
4. System displays a random challenge phrase.
5. User speaks the phrase.
6. Lip-sync verification confirms audio and mouth movement consistency.
7. Decision engine combines all results.
8. Access is granted only if all required checks pass.

---

## Example Authentication Session

```text
Face Verification      : PASS (0.92)
Voice Verification     : PASS (0.89)
Lip-Sync Verification  : PASS (0.94)

Final Result           : AUTHENTICATED
```

### Spoofing Attempt

```text
Face Verification      : PASS
Voice Verification     : PASS
Lip-Sync Verification  : FAIL

Final Result           : ACCESS DENIED
```

This prevents attackers from using:

* Recorded videos
* Deepfake videos
* Audio recordings
* Static photographs

---

## Project Structure

```text
TriGuard/
│
├── face/
│   ├── enrollment
│   ├── verification
│   └── embeddings
│
├── voice/
│   ├── enrollment
│   ├── verification
│   └── embeddings
│
├── lip_sync/
│   ├── challenge_response
│   └── verification
│
├── data/
│   ├── face_embeddings
│   ├── voice_embeddings
│   └── users
│
├── models/
│
├── configs/
│
└── main.py
```

---

## Technology Stack

### Computer Vision

* OpenCV
* MediaPipe
* DeepFace

### Deep Learning

* TensorFlow
* PyTorch
* SpeechBrain

### Speaker Recognition

* ECAPA-TDNN

### Backend

* Python

---

## Security Benefits

* Multi-modal biometric verification
* Resistance to replay attacks
* Challenge-response authentication
* Reduced false acceptance rates
* Improved identity assurance

---

## Future Improvements

* Mobile application support
* Liveness detection
* Anti-deepfake detection
* Continuous authentication
* Hardware security module integration
* Cloud-based deployment

---

## Author

Aryan Chougule

B.Tech Computer Science (AI & ML)

Computer Vision | Deep Learning | Biometrics | Edge AI
