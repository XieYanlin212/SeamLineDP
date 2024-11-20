This work is based on the environment
-python
-opencv
-torch
-macsac
Special Instructions：
The program corresponds to the article, except for additional new improvements
We set the end and start points of the stitching line in the overlapping area. We set the end and start points to the intersection of the overlapping areas of the two images. If you want to cancel this function 
    start_point = (int(intersection_A[1]), int(intersection_A[0]))
    end_point = (int(intersection_B[1]), int(intersection_B[0]))
change to
    end_point = np.unravel_index(np.argmin(dp[-1, :]), dp.shape)
    start_point = (int(end_point[0]), int(end_point[1]))

![image](https://github.com/user-attachments/assets/fd9825bb-cbdc-4472-a6ed-4b8fc9091815)
