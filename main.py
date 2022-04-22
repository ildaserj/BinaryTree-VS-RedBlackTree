###############     BINARY TREE VS RED BLACK TREE      ###############

import math
import random, pickle
from timeit import default_timer as timer
import numpy as np
from random import sample
import matplotlib.pyplot as plt


######################################################################
#                           BINARY TREE                              #


class TreeVertex:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None
        self.parent = None


class BinaryTree:
    def __init__(self):
        self.root = None

    def setRoot(self, x):
        self.root = x

    def getRoot(self):
        return self.root

    # ###   INSERT      #       #         INSERT

    def insert(self, value):
        z = TreeVertex(value)
        y = None
        x = self.root
        while x is not None:
            y = x
            if z.key <= x.key:
                x = x.left
                #print(value, "left BST")                                       #_________________________
            else:
                x = x.right
                #print(value, "right BST")
        z.parent = y
        if y is None:
            self.setRoot(z)
        elif z.key < y.key:
            y.left = z
        else:
            y.right = z

    # ###   INORDER TREE WALK      #      INORDER TREE WALK

    def inorderTreeWolk(self, x):
        if x is not None:
            self.inorderTreeWolk(x.left)
            print(x.key)
            self.inorderTreeWolk(x.right)

    # ###   FIND SUBTREE      #           FIND SUBTREE

    def findSubTree(self, x, key):
        while x is not None and x.key != key:
            if key < x.key:
                x = x.left
            else:
                x = x.right
        return x

    # ###          FIND      #      #     FIND

    def find(self, keyToFind):
        return self.findSubTree(self.root, keyToFind)

    # ###   PREORDER TREE WALK      #      PREORDER TREE WALK

    def preorderTreeWalk(self, x):
        if x is not None:
            print(x.key)
            self.inorderTreeWolk(x.left)
            self.inorderTreeWolk(x.right)

    # ###   MAX DEPTH       #      #       MAX DEPTH

    def maxDepth(self, x):
        if x is None:
            return 0
        else:
            leftHeight = self.maxDepth(x.left)
            rightHeight = self.maxDepth(x.right)
            return max(leftHeight, rightHeight) + 1


######################################################################
#                           RED BLACK TREE                           #

class BrVertex(TreeVertex):
    def __init__(self, key):
        super().__init__(key)
        self.color = None

    def setColor(self, col):
        if col == "R" or col == 1:
            self.color = 1
        if col == "B" or col == 0:
            self.color = 0
        else:
            print("Error color")

class RBTree:
    def __init__(self):
        self.NullVertex = BrVertex(None)
        self.NullVertex.color = 0
        self.root = self.NullVertex

    # ###   INORDER TREE WALK      #      INORDER TREE WALK

    def inorderTreeWalk(self, x):
        if x != self.NullVertex:
            self.inorderTreeWalk(x.left)
            self.inorderTreeWalk(x.right)

    # ###   PREORDER TREE WALK      #      PREORDER TREE WALK

    def preoderTreeWalk(self, x):
        if x != self.NullVertex:
            self.inorderTreeWalk(x.left)
            self.inorderTreeWalk(x.right)

    # ###           INSERT          #           INSERT

    def insert(self, k):
        z = TreeVertex(k)
        y = self.NullVertex
        x = self.root
        while x != self.NullVertex:
            y = x
            if z.key < y.key:
                x = x.left
                #print(k, "left RBT")
            else:
                x = x.right
                #print(k, "right RBT")
        z.parent = y
        if y == self.NullVertex:
            self.root = z
        elif z.key < y.key:
            y.left = z
        else:
            y.right = z
        z.left = self.NullVertex
        z.right = self.NullVertex
        z.color = 1
        self.insert_fixup(z)

    def insert_fixup(self, z):
        while z.parent.color == 1:
            if z.parent == z.parent.parent.left:
                y = z.parent.parent.right
                if y.color == 1:
                    z.parent.color = 0
                    y.color = 0
                    z.parent.parent.color = 1
                    z = z.parent.parent
                else:
                    if z == z.parent.right:
                        z = z.parent
                        self.leftRotate(z)
                    z.parent.color = 0
                    z.parent.parent.color = 1
                    self.rightRotate(z.parent.parent)
            else:
                y = z.parent.parent.left
                if y.color == 1:
                    z.parent.color = 0
                    y.color = 0
                    z.parent.parent.color = 1
                    z = z.parent.parent
                else:
                    if z == z.parent.left:
                        z = z.parent
                        self.rightRotate(z)
                    z.parent.color = 0
                    z.parent.parent.color = 1
                    self.leftRotate(z.parent.parent)
        self.root.color = 0

    def leftRotate(self, x):
        #print("left Rotate")
        y = x.right
        x.right = y.left
        if y.left != self.NullVertex:
            y.left.parent = x
        y.parent = x.parent
        if x.parent == self.NullVertex:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y
        y.left = x
        x.parent = y

    def rightRotate(self, y):
        #print("right rotate")
        x = y.left
        y.left = x.right
        if x.right != self.NullVertex:
            x.right.parent = y
        x.parent = y.parent
        if y.parent.key == self.NullVertex:
            self.root = x
        elif y == y.parent.left:
            y.parent.left = x
        else:
            y.parent.right = x
        x.right = y
        y.parent = x

    # ###   FIND SUBTREE      #           FIND SUBTREE
    def findSubTree(self, x, key):
        while x.key is not None and x.key != key:
            if key < x.key:
                x = x.left
            else:
                x = x.right
            return x

    # ###          FIND      #      #     FIND
    def find(self, keyToFind):
        return self.findSubTree(self.root, keyToFind)

    # ###   MAX DEPTH       #      #       MAX DEPTH
    def maxDepth(self, x):
        if x == self.NullVertex:
            return 0
        else:
            leftHeight = self.maxDepth(x.left)
            rightDepth = self.maxDepth(x.right)
            return max(leftHeight, rightDepth) + 1


#########################################################
#                     TEST START                        #
#########################################################
#                 vector costruction   ,                #
def random_vector(n):
    A = np.random.randint(0, n*n, size=n)
    print("random vector: ", A)
    return A


def random_vector_incr(n):
    A = np.random.randint(0, n*n, size=n)
    A.sort()
    print("random vector incr: ", A)
    return A


def random_vector_rev(n):
    A = []
    for i in range(n):
        x = random.randint(0, n)
        A.append(x)
    A.reverse()
    print("Random vector rev: ", A)
    return A


def BST_insert(T, A):
    start = timer()
    for i in range(0, len(A)):
        T.insert(A[i])
    end = timer()
    return end - start


def RBT_insert(T, A):
    start = timer()
    for i in range(0, len(A)):
        T.insert(A[i])
    end = timer()
    return end - start


def BST_search(T, num):
    Time = []
    start = timer()
    for i in range(0, len(num)):
        T.find(num[i])
    end = timer()
    Time.append(end - start)
    return np.mean(Time)


def RBT_search(T, num):
    Time = []
    start = timer()
    for i in range(0, len(num)):
        T.find(num[i])
    end = timer()
    Time.append(end - start)
    return np.mean(Time)


def testBST_random():
    Tins = []
    Ts = []
    Mins = []
    Ms = []
    dim = 10
    r = 0
    while r < 5:
        j = 0
        numberToSearch = random_vector(dim)
        while j < 5:
            A = random_vector(dim)
            BST_tree = BinaryTree()
            Tins.append(BST_insert(BST_tree, A))
            Ts.append(BST_search(BST_tree, numberToSearch))
            j += 1
        sumI = 0
        sumS = 0
        for k in range(0, len(Tins)):
            sumI += Tins[k]
            sumS += Ts[k]
        mediaI = sumI / len(Tins)
        mediaS = sumS / len(Ts)
        Mins.append(mediaI)
        Ms.append(mediaS)
        r += 1
        Tins[0:len(Tins)] = []
        Ts[0:len(Ts)] = []
        dim += 10                                             ###Dim
    pickle.dump(Mins, open("m1.p", "wb"))
    pickle.dump(Ms, open("m2.p", "wb"))

def testBST_search_random():
    T = []
    M = []
    dim = 10
    r = 0
    while r < 9000:
        j = 0
        numberToSearch = random_vector(dim)
        while j < 5:
            A = random_vector(dim)
            BST_tree = BinaryTree()
            BST_insert(BST_tree, A)

            T.append(BST_search(BST_tree, numberToSearch))
            j += 1
        sum = 0
        for k in range(0, len(T)):
            sum += T[k]
        media = sum / len(T)
        M.append(media)
        r += 2000
        T[0:len(T)] = []
        dim += 2000                                         #Dim
    pickle.dump(M, open("m2.p", "wb"))

def testBST_search_incr():
    T = []
    M = []
    BST_tree = BinaryTree()
    dim = 10
    r = 0
    while r < 9000:
        j = 0
        numberToSearch = random_vector(dim)
        while j < 5:
            A = random_vector_incr(dim)
            BST_insert(BST_tree, A)

            T.append(BST_search(BST_tree, numberToSearch))
            j += 1
        sum = 0
        for k in range(0, len(T)):
            sum += T[k]
        media = sum / len(T)
        M.append(media)
        r += 2000
        T[0:len(T)] = []
        dim += 2000                                             #Dim
    pickle.dump(M, open("m3.p","wb"))

def testBST_incr():
    Tins = []
    Ts = []
    Mins = []
    Ms = []
    dim = 10
    r = 0
    while r < 5:
        j = 0
        numberToSearch = random_vector(dim)
        while j < 5:
            A = random_vector_incr(dim)
            BRT_tree = BinaryTree()
            Tins.append(BST_insert(BRT_tree, A))
            Ts.append(BST_search(BRT_tree, numberToSearch))
            j += 1
        sumI = 0
        sumS = 0
        for k in range(0, len(Tins)):
            sumI += Tins[k]
            sumS = Ts[k]
        mediaI = sumI / len(Tins)
        mediaS = sumS / len(Ts)
        Mins.append(mediaI)
        Ms.append(mediaS)
        r += 1
        Tins[0:len(Tins)] = []
        Ts[0:len(Ts)] = []
        dim += 10                                        ## Dim
    pickle.dump(Mins, open("m5.p", "wb"))
    pickle.dump(Ms, open("m3.p", "wb"))

def testRBT_random():
    Tins = []
    Ts = []
    Mins = []
    Ms = []
    dim = 10
    r = 0
    while r < 5:
        j = 0
        numberToSearch = random_vector(dim)
        while j < 5:
            A = random_vector(dim)
            RBT_tree = RBTree()
            Tins.append(BST_insert(RBT_tree, A))
            Ts.append(BST_search(RBT_tree, numberToSearch))
            j += 1
        sumI = 0
        sumS = 0
        for k in range(0, len(Tins)):
            sumI += Tins[k]
            sumS = Ts[k]
        mediaI = sumI / len(Tins)
        mediaS = sumS / len(Ts)
        Mins.append(mediaI)
        Ms.append(mediaS)
        r += 1
        Tins[0:len(Tins)] = []
        Ts[0:len(Ts)] = []
        dim += 10  ## Dim

    pickle.dump(Mins, open("m6.p", "wb"))
    pickle.dump(Ms, open("m8.p", "wb"))



def testRBT_search_random():
    T = []
    M = []
    dim = 10
    r = 0
    while r < 9000:
        j = 0
        numberToSearch = random_vector(dim)
        while j < 5:
            A = random_vector(dim)
            RBT_tree = RBTree()

            T.append(RBT_search(RBT_tree, numberToSearch))
            j += 1
        sum = 0
        for k in range(0, len(T)):
            sum += T[k]
        media = sum / len(T)
        M.append(media)
        r += 2000
        T[0:len(T)] = []
        dim += 2000                                            ####Dim
    pickle.dump(M, open("m8.p", "wb"))

def testRBT_incr():

    Tins = []
    Ts = []
    Mins = []
    Ms = []
    dim = 10
    r = 0
    while r < 5:
        j = 0
        numberToSearch = random_vector(dim)
        while j < 5:
            A = random_vector_incr(dim)
            RBT_tree = RBTree()
            Tins.append(BST_insert(RBT_tree, A))
            Ts.append(BST_search(RBT_tree, numberToSearch))
            j += 1
        sumI = 0
        sumS = 0
        for k in range(0, len(Tins)):
            sumI += Tins[k]
            sumS = Ts[k]
        mediaI = sumI / len(Tins)
        mediaS = sumS / len(Ts)
        Mins.append(mediaI)
        Ms.append(mediaS)
        r += 1
        Tins[0:len(Tins)] = []
        Ts[0:len(Ts)] = []
        dim += 10  ## Dim
    pickle.dump(Mins, open("m7.p", "wb"))
    pickle.dump(Ms, open("m9.p", "wb"))


def testRBT_search_incr():
    T = []
    M = []
    RBT_tree = RBTree()
    dim = 10
    r = 0

    while r < 9000:
        j = 0
        numberToSearch = random_vector(dim)
        while j < 5:
            A = random_vector_incr(dim)
            RBT_insert(RBT_tree, A)

            T.append(RBT_search(RBT_tree, numberToSearch))
            j += 1
        sum = 0
        for k in range(0, len(T)):
            sum += T[k]
        media = sum / len(T)
        M.append(media)
        r += 2000
        T[0:len(T)] = []
        dim += 2000                                                ####Dim
    pickle.dump(M, open("m9.p", "wb"))

def RunAllTests():
    print("BST ins inc")
    testBST_incr()
    print("BST ins rand")
    testBST_random()
    print("RBT ins inc")
    testRBT_incr()
    print("RBT ins rand")
    testRBT_random()
   # print("BST serch inc")
   # testBST_search_incr()
   # print("BST search rand")
   # testBST_search_random()
   # print("RBT search inc")
   # testRBT_search_incr()
   # print("RBT search rand")
   # testRBT_search_random()


def PlotTests():
    RunAllTests()
    A = pickle.load(open("m1.p", "rb"))
    B = pickle.load(open("m6.p", "rb"))
    C = pickle.load(open("m5.p", "rb"))
    D = pickle.load(open("m7.p", "rb"))
    E = pickle.load(open("m2.p", "rb"))
    F = pickle.load(open("m3.p", "rb"))
    G = pickle.load(open("m8.p", "rb"))
    H = pickle.load(open("m9.p", "rb"))
    y = [10, 3000, 6000, 9000, 12000]
    plt.plot(y, A,  label="BST: inserimento elementi random")
    plt.plot(y, B,   label="RBT: inseriemnto elementi random")
    plt.legend()
    plt.xlabel("Numero elementi")
    plt.ylabel("Tempo esecuzione")
    plt.show()

    plt.plot(y, C, label="BST: inserimento elementi ordinati")
    plt.plot(y, D, label="RBT: inserimento elementi ordinati")
    plt.legend()
    plt.xlabel("Numero elementi")
    plt.ylabel("Tempo esecuzione")
    plt.show()

    plt.plot(y, E, label="BST: ricerca random")
    plt.plot(y, G, label="RBT: ricerca random")
    plt.legend()
    plt.xlabel("Numero Elementi")
    plt.ylabel("Tempo di Esecuzione")
    plt.show()

    plt.plot(y, F, label="BST: ricerca ordinata")
    plt.plot(y, H, label="RBT: ricerca ordinanta")
    plt.legend()
    plt.xlabel("Numero Elementi")
    plt.ylabel("Tempo di Esecuzione")
    plt.show()

def test_insert(File, repeat):
    BSTSearchGraph = []
    RBTGraph = []
    pickle_in = open(File, "rb")
    Set = pickle.load(pickle_in)
    Set1 = Set.copy()
    for j in range(1, len(Set)):
        print("Test: ", File, "Passo: ", j, "/", len(Set))
        R1 = []
        M1 = []
        T1 = BinaryTree()
        T2 = RBTree()
        for k in range(0, repeat):
            R1.append(BST_insert(T1, Set[j]))
            M1.append(RBT_insert(T2, Set1[j]))
        BSTSearchGraph.append(np.mean(R1))
        RBTGraph.append(np.mean(M1))
    Set = []
    ElementsNum = []
    pickle_in = open(File, "rb")
    Set = pickle.load(pickle_in)
    for z in range(1, len(Set)):
        A = Set[z]
        ElementsNum.append(len(A))
    plt.plot(ElementsNum, BSTSearchGraph)
    plt.plot(ElementsNum, RBTGraph)
    plt.xlabel('Numero di elementi')
    plt.ylabel('Tempo di esecuzione')
    plt.title('Inserimento: Albero Binario vs Albero RN')
    plt.legend(['Albero binario', 'Albero RN'])
    plt.show()


if __name__ == '__main__':
    PlotTests()
    #rb = RBTree()
    #binary = BinaryTree()
    #vector = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    #RBT_insert(rb, vector)
    #BST_insert(binary, vector)
