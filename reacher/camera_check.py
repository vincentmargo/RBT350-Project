def main():
    import os
    import glob

    print("=== WSL camera visibility check ===")
    vids = sorted(glob.glob("/dev/video*"))
    if not vids:
        print("No /dev/video* devices found.")
        print("If you're on WSL2, your webcam is likely not exposed to Linux.")
        print("Workarounds: run on Windows Python, or configure webcam passthrough.")
        return

    print("Found video devices:")
    for p in vids:
        try:
            st = os.stat(p)
            print(f"- {p} (mode={oct(st.st_mode)})")
        except Exception:
            print(f"- {p}")

    try:
        import cv2
    except Exception as e:
        print("\nOpenCV (cv2) not available in this Python environment.")
        print(f"Import error: {e}")
        print("Activate your project environment (e.g., conda env) and retry.")
        return

    print("\nTrying to open camera indices 0..5 (default settings)")
    backend = getattr(cv2, "CAP_V4L2", 0)
    for idx in range(6):
        cap = cv2.VideoCapture(idx, backend) if backend else cv2.VideoCapture(idx)
        ok = cap.isOpened()
        print(f"- index {idx}: {'OK' if ok else 'FAIL'}")
        if ok:
            ret, frame = cap.read()
            print(f"  read frame: {'OK' if ret and frame is not None else 'FAIL'} shape={getattr(frame, 'shape', None)}")
        cap.release()

    print("\nTrying index 0 with MJPG 640x480 @ 15fps (common workaround)")
    cap = cv2.VideoCapture(0, backend) if backend else cv2.VideoCapture(0)
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok_any = False
        for i in range(20):
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"  warmup read {i}: OK shape={frame.shape}")
                ok_any = True
                break
        if not ok_any:
            print("  MJPG warmup reads: FAIL (timeout/no frames)")
    else:
        print("  Could not open index 0 for MJPG test.")
    cap.release()


if __name__ == "__main__":
    main()

