import numpy as np
from skimage.measure import regionprops, label
A = np.array(
    [[-1 , -1 , -1 , -1 , -1 , -1 , -1 , -1 , -1 , -1 , -1],
     [-1 ,  0 ,  0 ,  0 ,  0 , -1 ,  0 , -1 ,  0 ,  0 , -1],
     [-1 ,  0 ,  0 ,  0 ,  0 , -1 ,  0 , -1 ,  0 ,  0 , -1],
     [-1 ,  0 ,  0 ,  0 ,  0 , -1 ,  0 , -1 ,  0 ,  0 , -1],
     [-1 ,  0 ,  0 ,  0 ,  0 , -1 ,  0 , -1 ,  0 ,  0 , -1],
     [-1 ,  0 ,  0 ,  0 ,  0 , -1 ,  0 , -1 ,  0 ,  0 , -1],
     [-1 ,  0 ,  0 ,  0 ,  0 , -1 ,  0 , -1 ,  0 ,  0 , -1],
     [-1 ,  0 ,  0 ,  0 ,  0 , -1 ,  0 , -1 ,  0 ,  0 , -1],
     [-1 ,  0 ,  0 ,  0 ,  0 , -1 ,  0 , -1 ,  0 ,  0 , -1],
     [-1 ,  0 ,  0 ,  0 ,  0 ,  0,  0 , -1 ,  0 ,  0 , -1],
     [-1 , -1 , -1 , -1 , -1 , -1 , -1 , -1 , -1 , -1 , -1],
     ])
map = np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -1],
 [-1, 0, -1, 0, -1, 1, -1, 0, -1, 1, -1, 0, -1, 1, -1, 0, -1],
 [-1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, -1],
 [-1, 0, -1, 1, -1, 1, -1, 0, -1, 0, -1, 0, -1, 0, -1, 1, -1],
 [-1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, -1],
 [-1, 0, -1, 0, -1, 1, -1, 0, -1, 1, -1, 0, -1, 1, -1, 1, -1],
 [-1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, -1],
 [-1, 0, -1, 1, -1, 0, -1, 1, -1, 1, -1, 0, -1, 0, -1, 0, -1],
 [-1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, -1],
 [-1, 0, -1, 1, -1, 1, -1, 0, -1, 0, -1, 0, -1, 0, -1, 1, -1],
 [-1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, -1],
 [-1, 1, -1, 0, -1, 1, -1, 0, -1, 1, -1, 0, -1, 0, -1, 0, -1],
 [-1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, -1],
 [-1, 0, -1, 1, -1, 0, -1, 1, -1, 0, -1, 0, -1, 0, -1, 0, -1],
 [-1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, -1],
 [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])
map_without_crates = np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
 [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
 [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1],
 [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
 [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1],
 [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
 [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1],
 [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
 [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1],
 [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0, 0,  0,  0,  0,  0,  0, -1],
 [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1],
 [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
 [-1,  0, -1,  0, -1,  0, -1,  0, -1, 0, -1,  0, -1,  0, -1,  0, -1],
 [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
 [-1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1,  0, -1],
 [-1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1],
 [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])

def test_connected():
    p1 = (1, 1)  # point upper left
    # p2 = (3, 4)  # point lower right... CONNECTED
    # p2 = (6, 6)  # point lower right... CONNECTED
    p2 = (6, 9)  # point lower right... CONNECTED

    print(A)
    B = np.zeros_like(A)
    B[A==A[p1[0],p1[1]]] = 1
    print(B)
    print('Connected points?')
    # C = B[p1[0]:p2[0],p1[1]:p2[1]]
    # print(C)
    # print('sum(C', np.sum(C))
    connected = False
    labeled = label(B, connectivity=1)
    print(labeled)
    connected = labeled[p1[0],p1[1]] == labeled[p2[0],p2[1]]
    if connected:
        print('YES')
    else:
        print('NO')

    print('Test map....')
    print(map)
    p3 = (15, 15)
    map[map == 1] = -1
    map[map == map[p3[0], p3[1]]] = 1
    map[map != 1] = 0

    labeled_map = label(map, connectivity=1)
    print("Labeled map:")
    print(labeled_map)

    labeled_value = labeled_map[p3]
    print(labeled_value)

    num = sum(labeled_map[labeled_map == labeled_value])
    print("Number of free tiles: {}".format(num))

    print('Test map without crates....')
    print(labeled_map)
    p3 = (15, 15)
    map_without_crates[map_without_crates == 1] = -1
    map_without_crates[map_without_crates == map_without_crates[p3[0], p3[1]]] = 1
    map_without_crates[map_without_crates != 1] = 0

    labeled_map_without_crates = label(map_without_crates, connectivity=1)
    # print("Labeled map_without_crates:")
    # print(labeled_map_without_crates)

    labeled_value = labeled_map_without_crates[p3]
    # print(labeled_value)

    num = sum(labeled_map_without_crates[labeled_map_without_crates == labeled_value])
    # print("Number of free tiles: {}".format(num))

    bombs = [(1, 15)]
    for b in bombs:
        # print("Pos bomb: ", labeled_map[b[0], b[1]])
        labeled_map[b[0], b[1]] = -1
    # print("Include bombs")
    # print(labeled_map)


if __name__ == '__main__':
    # test_suit_case()
    test_connected()