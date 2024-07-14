import React, { useRef, useEffect, useState, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as poseDetection from '@tensorflow-models/pose-detection';

const creatures = {
  blob: {
    body: (ctx, keypoints) => {
      const neck = keypoints.find((kp) => kp.name === "neck");
      const leftHip = keypoints.find((kp) => kp.name === "left_hip");
      const rightHip = keypoints.find((kp) => kp.name === "right_hip");

      if (neck && leftHip && rightHip) {
        ctx.beginPath();
        ctx.moveTo(neck.x, neck.y);
        ctx.quadraticCurveTo(
          (leftHip.x + rightHip.x) / 2,
          (leftHip.y + rightHip.y) / 2,
          (leftHip.x + rightHip.x) / 2,
          Math.max(leftHip.y, rightHip.y)
        );
        ctx.quadraticCurveTo(
          (leftHip.x + rightHip.x) / 2,
          (leftHip.y + rightHip.y) / 2,
          neck.x,
          neck.y
        );
        ctx.fillStyle = "rgba(0, 255, 0, 0.5)";
        ctx.fill();
      }
    },
    eyes: (ctx, keypoints) => {
      const leftEye = keypoints.find((kp) => kp.name === "left_eye");
      const rightEye = keypoints.find((kp) => kp.name === "right_eye");

      if (leftEye && rightEye) {
        ctx.beginPath();
        ctx.arc(leftEye.x, leftEye.y, 10, 0, 2 * Math.PI);
        ctx.arc(rightEye.x, rightEye.y, 10, 0, 2 * Math.PI);
        ctx.fillStyle = "white";
        ctx.fill();
        ctx.beginPath();
        ctx.arc(leftEye.x, leftEye.y, 5, 0, 2 * Math.PI);
        ctx.arc(rightEye.x, rightEye.y, 5, 0, 2 * Math.PI);
        ctx.fillStyle = "black";
        ctx.fill();
      }
    },
    limbs: (ctx, keypoints) => {
      const limbs = [
        ["left_shoulder", "left_elbow", "left_wrist"],
        ["right_shoulder", "right_elbow", "right_wrist"],
        ["left_hip", "left_knee", "left_ankle"],
        ["right_hip", "right_knee", "right_ankle"],
      ];

      limbs.forEach((limb) => {
        const points = limb
          .map((name) => keypoints.find((kp) => kp.name === name))
          .filter(Boolean);
        if (points.length === 3) {
          ctx.beginPath();
          ctx.moveTo(points[0].x, points[0].y);
          ctx.quadraticCurveTo(
            points[1].x,
            points[1].y,
            points[2].x,
            points[2].y
          );
          ctx.lineWidth = 10;
          ctx.strokeStyle = "rgba(0, 200, 0, 0.7)";
          ctx.stroke();
        }
      });
    },
  },
  // Add more creatures here
};

const PoseEstimation = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const [detector, setDetector] = useState(null);
  const [isVideoReady, setIsVideoReady] = useState(false);
  const [debugMode, setDebugMode] = useState(false);
  const [canvasSize, setCanvasSize] = useState({ width: 640, height: 480 });
  const [currentCreature, setCurrentCreature] = useState("blob");

  const updateCanvasSize = useCallback(() => {
    if (containerRef.current) {
      const width = containerRef.current.clientWidth;
      const height = window.innerHeight - 100; // Leave 100px for title and settings
      setCanvasSize({ width, height });
    }
  }, []);

  useEffect(() => {
    updateCanvasSize();
    window.addEventListener('resize', updateCanvasSize);
    return () => window.removeEventListener('resize', updateCanvasSize);
  }, [updateCanvasSize]);

  useEffect(() => {
    const initializeTF = async () => {
      await tf.ready();
      await tf.setBackend('webgl');

      const detectorConfig = {modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING};
      const detector = await poseDetection.createDetector(poseDetection.SupportedModels.MoveNet, detectorConfig);
      setDetector(detector);
    };

    initializeTF();
  }, []);

  useEffect(() => {
    const setupCamera = async () => {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      const video = videoRef.current;
      if (video) {
        video.srcObject = stream;
        video.onloadedmetadata = () => {
          video.play();
          setIsVideoReady(true);
        };
      }
    };

    setupCamera();
  }, []);

  useEffect(() => {
    if (detector && videoRef.current && canvasRef.current && isVideoReady) {
      const detectPose = async () => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.save();
        ctx.scale(-1, 1);
        ctx.translate(-canvas.width, 0);

        if (debugMode) {
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        }

        const poses = await detector.estimatePoses(video);

        poses.forEach(pose => {
          pose.keypoints.forEach(keypoint => {
            const x = (keypoint.x / video.videoWidth) * canvas.width;
            const y = (keypoint.y / video.videoHeight) * canvas.height;
            ctx.beginPath();
            ctx.arc(x, y, 5, 0, 2 * Math.PI);
            ctx.fillStyle = 'red';
            ctx.fill();
          });
        });

        ctx.restore();
        requestAnimationFrame(detectPose);
      };

      detectPose();
    }
  }, [detector, isVideoReady, debugMode, canvasSize]);

  useEffect(() => {
    if (detector && videoRef.current && canvasRef.current && isVideoReady) {
      const detectPose = async () => {
        const video = videoRef.current;
        const canvas = canvasRef.current;
        const ctx = canvas.getContext("2d");

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.save();
        ctx.scale(-1, 1);
        ctx.translate(-canvas.width, 0);

        if (debugMode) {
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        }

        const poses = await detector.estimatePoses(video);

        if (poses.length > 0) {
          const keypoints = poses[0].keypoints;
          const creature = creatures[currentCreature];

          creature.body(ctx, keypoints);
          creature.limbs(ctx, keypoints);
          creature.eyes(ctx, keypoints);
        }

        ctx.restore();
        requestAnimationFrame(detectPose);
      };

      detectPose();
    }
  }, [detector, isVideoReady, debugMode, canvasSize, currentCreature]);

  const toggleDebugMode = () => {
    setDebugMode(!debugMode);
  };

  const changeCreature = () => {
    const creatureNames = Object.keys(creatures);
    const currentIndex = creatureNames.indexOf(currentCreature);
    const nextIndex = (currentIndex + 1) % creatureNames.length;
    setCurrentCreature(creatureNames[nextIndex]);
  };

  return (
    <div
      ref={containerRef}
      style={{
        width: "100%",
        height: "100vh",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <h1 style={{ textAlign: "center", margin: "10px 0" }}>
        Pose Estimation Creatures
      </h1>
      <div style={{ flex: 1, position: "relative" }}>
        <video
          ref={videoRef}
          style={{ display: "none" }}
          width={canvasSize.width}
          height={canvasSize.height}
        />
        <canvas
          ref={canvasRef}
          width={canvasSize.width}
          height={canvasSize.height}
          style={{
            border: "1px solid black",
            position: "absolute",
            top: 0,
            left: 0,
            width: "100%",
            height: "100%",
          }}
        />
      </div>
      <div style={{ padding: "10px", textAlign: "center" }}>
        <button onClick={toggleDebugMode} style={{ marginRight: "10px" }}>
          {debugMode ? "Disable Debug Mode" : "Enable Debug Mode"}
        </button>
        <button onClick={changeCreature}>Change Creature</button>
      </div>
    </div>
  );
};

export default PoseEstimation;
