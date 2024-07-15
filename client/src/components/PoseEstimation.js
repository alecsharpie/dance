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
  const [multiPoseMode, setMultiPoseMode] = useState(false);
  const [isChangingMode, setIsChangingMode] = useState(false);

  const updateCanvasSize = useCallback(() => {
    if (containerRef.current) {
      const width = containerRef.current.clientWidth;
      const height = window.innerHeight - 100;
      setCanvasSize({ width, height });
    }
  }, []);

  const createDetector = useCallback(async () => {
    setIsChangingMode(true);
    if (detector) {
      await detector.dispose();
    }
    try {
      const detectorConfig = {
        modelType: multiPoseMode
          ? poseDetection.movenet.modelType.MULTIPOSE_LIGHTNING
          : poseDetection.movenet.modelType.SINGLEPOSE_LIGHTNING,
      };
      const newDetector = await poseDetection.createDetector(
        poseDetection.SupportedModels.MoveNet,
        detectorConfig
      );
      setDetector(newDetector);
    } catch (error) {
      console.error("Error creating detector:", error);
    } finally {
      setIsChangingMode(false);
    }
    // get a warning that detector should be included here
    // ignore this, as it would be a circular dependency
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [multiPoseMode]);

  useEffect(() => {
    updateCanvasSize();
    window.addEventListener("resize", updateCanvasSize);
    return () => window.removeEventListener("resize", updateCanvasSize);
  }, [updateCanvasSize]);

  useEffect(() => {
    const initializeTF = async () => {
      await tf.ready();
      await tf.setBackend("webgl");
      createDetector();
    };

    initializeTF();
  }, [createDetector]);

  useEffect(() => {
    if (!isChangingMode) {
      createDetector();
    }
  }, [multiPoseMode, createDetector, isChangingMode]);

  useEffect(() => {
    const setupCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: true,
        });
        const video = videoRef.current;
        if (video) {
          video.srcObject = stream;
          video.onloadedmetadata = () => {
            video.play();
            setIsVideoReady(true);
          };
        }
      } catch (error) {
        console.error("Error setting up camera:", error);
      }
    };

    setupCamera();
  }, []);

  useEffect(() => {
    let animationFrameId;
    let isDetecting = false;

    const detectPose = async () => {
      if (
        isChangingMode ||
        !detector ||
        !videoRef.current ||
        !canvasRef.current ||
        !isVideoReady ||
        isDetecting
      ) {
        animationFrameId = requestAnimationFrame(detectPose);
        return;
      }

      isDetecting = true;
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const ctx = canvas.getContext("2d");

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (debugMode) {
        ctx.save();
        ctx.scale(-1, 1);
        ctx.drawImage(video, -canvas.width, 0, canvas.width, canvas.height);
        ctx.restore();
      }

      try {
        const poses = await detector.estimatePoses(video);

        poses.forEach((pose, index) => {
          const keypoints = pose.keypoints.map((keypoint) => ({
            ...keypoint,
            x: canvas.width - (keypoint.x / video.videoWidth) * canvas.width,
            y: (keypoint.y / video.videoHeight) * canvas.height,
          }));

          const creature = creatures[currentCreature];
          const hue = multiPoseMode ? (index * 137) % 360 : 120;
          ctx.strokeStyle = `hsla(${hue}, 100%, 50%, 0.7)`;

          creature.limbs(ctx, keypoints);
          creature.eyes(ctx, keypoints);

          if (debugMode) {
            keypoints.forEach((keypoint) => {
              ctx.beginPath();
              ctx.arc(keypoint.x, keypoint.y, 5, 0, 2 * Math.PI);
              ctx.fillStyle = `hsl(${hue}, 100%, 50%)`;
              ctx.fill();
              ctx.fillStyle = "white";
              ctx.fillText(keypoint.name, keypoint.x + 5, keypoint.y - 5);
            });
          }
        });
      } catch (error) {
        console.error("Error in pose estimation:", error);
      }

      isDetecting = false;
      animationFrameId = requestAnimationFrame(detectPose);
    };

    detectPose();

    return () => {
      if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
      }
    };
  }, [
    detector,
    isVideoReady,
    debugMode,
    canvasSize,
    currentCreature,
    multiPoseMode,
    isChangingMode,
  ]);

  const toggleDebugMode = useCallback(() => setDebugMode((prev) => !prev), []);

  const changeCreature = useCallback(() => {
    const creatureNames = Object.keys(creatures);
    const currentIndex = creatureNames.indexOf(currentCreature);
    const nextIndex = (currentIndex + 1) % creatureNames.length;
    setCurrentCreature(creatureNames[nextIndex]);
  }, [currentCreature]);

  const toggleMultiPoseMode = useCallback(() => {
    if (!isChangingMode) {
      setMultiPoseMode((prev) => !prev);
    }
  }, [isChangingMode]);

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
        <button onClick={changeCreature} style={{ marginRight: "10px" }}>
          Change Creature
        </button>
        <button onClick={toggleMultiPoseMode} disabled={isChangingMode}>
          {isChangingMode
            ? "Changing Mode..."
            : multiPoseMode
            ? "Single Person Mode"
            : "Multiple People Mode"}
        </button>
      </div>
    </div>
  );
};

export default PoseEstimation;
