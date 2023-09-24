import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # hsv form
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # lower and upper bounds
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    lower_purple = np.array([120, 50, 50])
    upper_purple = np.array([160, 255, 255])

    # masks for isolating blue and purple regions
    blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
    purple_mask = cv2.inRange(hsv, lower_purple, upper_purple)

    # Combine both masks
    combined_mask = cv2.bitwise_or(blue_mask, purple_mask)

    # Gaussian blur combine mask
    blurred_mask = cv2.GaussianBlur(combined_mask, (5, 5), 0)

    # circles in filtered mask
    circles = cv2.HoughCircles(
        blurred_mask, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
        param1=50, param2=30, minRadius=15, maxRadius=100)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for circle in circles[0, :]:
            center = (circle[0], circle[1])
            radius = circle[2]
            # diameter = 2 * radius
            # Draw the circle and its diameter on the original frame
            cv2.circle(frame, center, radius, (0, 255, 0), 2)
            # cv2.line(frame, (circle[0] - radius, circle[1]), (circle[0] + radius, circle[1]), (0, 255, 0), 2)

    cv2.imshow("Blue and Purple Circle Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
