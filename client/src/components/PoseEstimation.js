import React, { useRef, useEffect, useState, useCallback } from "react";
import * as tf from "@tensorflow/tfjs";
import * as poseDetection from "@tensorflow-models/pose-detection";

const creatures = {
  blob: {
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
    window.addEventListener("resize", updateCanvasSize);
    return () => window.removeEventListener("resize", updateCanvasSize);
  }, [updateCanvasSize]);

  useEffect(() => {
    const initializeTF = async () => {
      await tf.ready();
      await tf.setBackend("webgl");

      const detectorConfig = {
        modelType: poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
      };
      const detector = await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        detectorConfig
      );
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
        const ctx = canvas.getContext("2d");

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (debugMode) {
          // Draw mirrored video
          ctx.save();
          ctx.scale(-1, 1);
          ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
          ctx.restore();
        }

        const poses = await detector.estimatePoses(video);

        if (poses.length > 0) {
          const keypoints = poses[0].keypoints.map((keypoint) => ({
            ...keypoint,
            x: canvas.width - (keypoint.x / video.videoWidth) * canvas.width, // Mirror X coordinate
            y: (keypoint.y / video.videoHeight) * canvas.height,
          }));

          const creature = creatures[currentCreature];

          creature.limbs(ctx, keypoints);
          creature.eyes(ctx, keypoints);

          // Debug: Draw keypoints
          if (debugMode) {
            keypoints.forEach((keypoint) => {
              ctx.beginPath();
              ctx.arc(keypoint.x, keypoint.y, 5, 0, 2 * Math.PI);
              ctx.fillStyle = "blue";
              ctx.fill();
              ctx.fillStyle = "white";
              ctx.fillText(keypoint.name, keypoint.x + 5, keypoint.y - 5);
            });
          }
        }

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
