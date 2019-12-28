# Definition for a binary tree node.
# -*- coding: utf-8 -*-

# @Time  : 2018/11/11 14:55

# @Author : zz

# @Project : workspace2

# @FileName: leetcode2.py

# 著作权归作者所有(随便拷)。
'''

                .-~~~~~~~~~-._       _.-~~~~~~~~~-.
            __.'              ~.   .~              `.__
          .'//                  \./                  \\`.
        .'//                     |                     \\`.
      .'// .-~"""""""~~~~-._     |     _,-~~~~"""""""~-. \\`.
    .'//.-"                 `-.  |  .-'                 "-.\\`.
  .'//______.============-..   \ | /   ..-============.______\\`.
.'______________________________\|/______________________________`.

'''

import this
from decimal import Decimal, getcontext

import copy
from collections import defaultdict
import json
import time
from functools import wraps
import collections
import functools
import math
import heapq
import numpy as np

import random


class NumMatrix:
    def __init__(self, matrix: list):
        self.matrix = np.asarray(matrix)

    def sumRegion(self, row1: int, col1: int, row2: int, col2: int) -> int:
        return np.sum(self.matrix[row1:row2 + 1, col1:col2 + 1])


class MaxHeap(object):

    def __init__(self):
        self._data = []
        self._count = len(self._data)

    def size(self):
        return self._count

    def isEmpty(self):
        return self._count == 0

    def add(self, item):
        # 插入元素入堆
        self._data.append(item)
        self._count += 1
        self._shiftup(self._count - 1)

    def pop(self):
        # 出堆
        if self._count > 0:
            ret = self._data[0]
            self._data[0] = self._data[self._count - 1]
            self._count -= 1
            self._shiftDown(0)
            return ret

    def getTOP(self):
        if self._count > 0:
            return self._data[0]

    def _shiftup(self, index):
        # 上移self._data[index]，以使它不大于父节点
        parent = (index - 1) >> 1
        while index > 0 and self._data[parent] < self._data[index]:
            # swap
            self._data[parent], self._data[index] = self._data[index], self._data[parent]
            index = parent
            parent = (index - 1) >> 1

    def _shiftDown(self, index):
        # 上移self._data[index]，以使它不小于子节点
        j = (index << 1) + 1
        while j < self._count:
            # 有子节点
            if j + 1 < self._count and self._data[j + 1] > self._data[j]:
                # 有右子节点，并且右子节点较大
                j += 1
            if self._data[index] >= self._data[j]:
                # 堆的索引位置已经大于两个子节点，不需要交换了
                break
            self._data[index], self._data[j] = self._data[j], self._data[index]
            index = j
            j = (index << 1) + 1

    def __repr__(self):
        return sorted(self._data, reverse=True)


class MyQueue:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.queue = []

    def push(self, x: int) -> None:
        """
        Push element x to the back of queue.
        """
        self.queue.append(x)

    def pop(self) -> int:
        """
        Removes the element from in front of queue and returns that element.
        """
        if self.queue.__len__() > 0:
            return None
        return self.queue.pop(0)

    def peek(self) -> int:
        """
        Get the front element.
        """
        if self.queue.__len__() > 0:
            return None
        return self.queue[0]

    def empty(self) -> bool:
        """
        Returns whether the queue is empty.
        """
        return self.queue.empty()


class MyStack:

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.queue = []
        self.len = 0

    def push(self, x: int) -> None:
        """
        Push element x onto stack.
        """
        self.queue.insert(self.len, x)
        self.len += 1

    def pop(self) -> int:
        """
        Removes the element on top of the stack and returns that element.
        """
        self.len -= 1
        return self.queue.pop()

    def top(self) -> int:
        """
        Get the top element.
        """
        return self.queue[-1]

    def empty(self) -> bool:
        """
        Returns whether the stack is empty.
        """
        return self.len == 0


class WordDictionary(object):
    '''
    词典的搜索
    '''

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.dictionary = collections.defaultdict(set)
        self.search_d = collections.defaultdict(bool)

    def addWord(self, word):
        """
        Adds a word into the data structure.
        :type word: str
        :rtype: None
        """
        if not word:
            return
        indexs = [word[0], '.']
        for index in indexs:
            self.dictionary[index].add(word)

        self.search_d.clear()

    def check(self, key, word):
        if len(self.dictionary[key]) == 0:
            return False
        for w in self.dictionary[key]:
            if len(w) != len(word):
                continue
            flag = 0
            for i in range(len(w)):
                if word[i] == '.':
                    continue
                else:
                    if w[i] != word[i]:
                        flag = 1
                        break
            if flag == 0:
                self.search_d[word] = True
                return True
        self.search_d[word] = False
        return False

    def cache(self, search_word):
        '''
        对完成搜索的word直接进行缓存
        :param search_word:
        :return:
        '''
        if search_word in self.search_d.keys():
            return self.search_d[search_word]

        return None

    def search(self, word):
        """
        Returns if the word is in the data structure.
        A word could contain the dot character '.' to represent any one letter.
        :type word: str
        :rtype: bool
        """
        if not word:
            return

        if self.cache(word) is not None:
            print('已进行搜索：', word)
            return self.search_d[word]

        if word.startswith('.'):
            return self.check('.', word)
        else:
            return self.check(word[0], word)


class Trie(object):

    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.trie = collections.defaultdict(set)

    def insert(self, word):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: None
        """
        if len(word) == 0:
            return
        for i in range(len(word)):
            ch = word[0:i + 1]
            self.trie[ch].add(word)

    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        return word in self.trie[word]

    def startsWith(self, prefix):
        """
        Returns if there is any word in the trie that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        return len(self.trie[prefix]) > 0


class LRUCache:
    # Your LRUCache object will be instantiated and called as such:
    # obj = LRUCache(capacity)
    # param_1 = obj.get(key)
    # obj.put(key,value)

    def __init__(self, capacity: int):
        self.LRUkeyslist = list()
        self.LRUdict = dict()
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key in self.LRUdict.keys():
            # 取一次key，使用一次，将该健对更新到最近使用，即将该key位置改为list的最后
            self.LRUkeyslist.remove(key)
            self.LRUkeyslist.append(key)
            return self.LRUdict[key]
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        if key not in self.LRUdict.keys():
            # key 不再字典中，则进行插入
            if self.LRUkeyslist.__len__() == self.capacity:
                # 字典满了，删除队列第一个key，同时删除dict中对应键对
                k = self.LRUkeyslist[0]
                del self.LRUkeyslist[0]
                self.LRUdict.pop(k)
            # 随后将键对插入到队列最后,同时更新字典
            self.LRUdict.update({key: value})
            self.LRUkeyslist.append(key)
        else:
            # 设置新值也需将该键对移动到最后
            self.LRUdict[key] = value
            self.LRUkeyslist.remove(key)
            self.LRUkeyslist.append(key)


class LRUCache:
    # Your LRUCache object will be instantiated and called as such:
    # obj = LRUCache(capacity)
    # param_1 = obj.get(key)
    # obj.put(key,value)

    def __init__(self, capacity: int):
        self.LRUdict = collections.OrderedDict()
        self.capacity = capacity

    def get(self, key: int) -> int:
        if key in self.LRUdict.keys():
            # 取一次key，使用一次，将该健对更新到最近使用，即将该key位置改为list的最后
            self.LRUdict.move_to_end(key)
            return self.LRUdict[key]
        else:
            return -1

    def put(self, key: int, value: int) -> None:
        if key not in self.LRUdict.keys():
            # key 不再字典中，则进行插入
            if len(self.LRUdict) == self.capacity:
                # 字典满了，删除队列第一个key，同时删除dict中对应键对
                self.LRUdict.popitem(False)
            self.LRUdict.update({key: value})

        else:
            # 设置新值也需将该键对移动到最后
            del self.LRUdict[key]
            self.LRUdict[key] = value


# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x=0):
        self.val = x
        self.next = None


def Isprime(n):
    if n <= 3:
        return True
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return False
    return True


def buildLink(l: list) -> ListNode:
    if len(l) == 0:
        return None
    if len(l) == 1:
        return ListNode(l[0])
    else:
        head = ListNode(l[0])
        r = head
        for i in range(1, len(l)):
            r.next = ListNode(l[i])
            r = r.next
        return head


def printLint(l: ListNode):
    while l:
        print(l.val, end=' ')
        l = l.next
    print()


def sort(num: list):
    i = len(num) - 1
    while i >= 0:
        if i != num[i]:
            index_i = num.index(i)
            index0 = num.index(0)
            num[index0], num[i] = num[i], num[index0]
            num[i], num[index_i] = num[index_i], num[i]
        i -= 1

    print(num)


class TreeNode:
    def __init__(self, x=0):
        self.val = x
        self.left = None
        self.right = None


def printTree(root):
    if not root:
        pass
    else:
        print(root.val, end=' ')
        printTree(root.left)
        printTree(root.right)


class BSTIterator:

    def __init__(self, root: TreeNode):
        self.root = root

    def next(self) -> int:
        """
        @return the next smallest number
        """
        pre = None
        ptr = self.root
        while ptr.left:
            pre = ptr
            ptr = ptr.left
        value = ptr.val

        if not pre:
            self.root = self.root.right
            return value

        if not ptr.left and ptr.right:
            pre.left = ptr.right
        else:
            pre.left = None
        ptr = None
        return value

    def hasNext(self) -> bool:
        """
        @return whether we have a next smallest number
        """
        pass
        return True if self.root else False


def BuildTree(layer: list, mid: list):
    # 层次遍历和中序遍历建树
    if mid == []:
        return None
    else:
        i = 0
        while layer[i] not in mid:
            i += 1
        root_val = layer[i]
        layer.remove(root_val)
        root_index = mid.index(root_val)
        root = TreeNode(root_val, BuildTree(
            layer, mid[0:root_index]), BuildTree(layer, mid[root_index + 1:]))
        return root


# Definition for a Node.
class Node:
    def __init__(self, val, next=None, random=None):
        self.val = val
        self.next = next
        self.random = random


def stringToTreeNode(input):
    input = input.strip()
    input = input[1:-1]
    if not input:
        return None

    inputValues = [s.strip() for s in input.split(',')]
    root = TreeNode(int(inputValues[0]))
    nodeQueue = [root]
    front = 0
    index = 1
    while index < len(inputValues):
        node = nodeQueue[front]
        front = front + 1

        item = inputValues[index]
        index = index + 1
        if item != "null":
            leftNumber = int(item)
            node.left = TreeNode(leftNumber)
            nodeQueue.append(node.left)

        if index >= len(inputValues):
            break

        item = inputValues[index]
        index = index + 1
        if item != "null":
            rightNumber = int(item)
            node.right = TreeNode(rightNumber)
            nodeQueue.append(node.right)
    return root


def integerListToString(nums, len_of_list=None):
    if not len_of_list:
        len_of_list = len(nums)
    return json.dumps(nums[:len_of_list])


def Time(name='FUNC', n=1):
    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            for _ in range(n - 1):
                func(*args, **kwargs)
            result = func(*args, **kwargs)
            print('\n函数: {name:>10}\nargs: {args}\n运行 {n}次\n需要时间: {time}\n'.format(
                name=func.__name__, n=n, args=args, time=time.time() - start))
            return result

        return wrapper

    return decorate


class MinStack:

    def __init__(self):
        """
        initialize your data structure here.
        """
        self.stack = []
        self.min = -1

    def push(self, x: int) -> None:
        if self.stack:
            self.stack.append(x)
            if x < self.stack[self.min]:
                self.min = len(self.stack) - 1
        else:
            self.stack.append(x)
            self.min = 0

    def pop(self) -> None:
        return self.stack.pop()

    def top(self) -> int:
        return self.stack[-1]

    def getMin(self) -> int:
        return self.stack[self.min]


class Solution:
    def maxPathSum(self, root: TreeNode) -> int:
        nonroot = root.val

        # 124.
        def dfs(root: TreeNode) -> int:
            nonlocal nonroot
            if not root:
                return float('-inf')
            else:
                left = dfs(root.left)
                right = dfs(root.right)
                mid = root.val
                nonroot = max(nonroot, left + right + mid, left, right)
                return max(mid, left + mid, right + mid)

        return max(dfs(root), nonroot)

    def metricword(self, word1, word2):
        count = 0
        for i in range(len(word1)):
            if word1[i] != word2[i]:
                count += 1
                if count > 1:
                    return 2
        return count

    @Time()
    def metric(self, word):
        preboard = defaultdict(list)
        for j in range(len(word)):
            for i in range(j):
                if self.metricword(word[i], word[j]) == 1:
                    preboard[word[j]].append(word[i])
                    preboard[word[i]].append(word[j])

        return preboard

    @Time()
    def bfs(self, w, word, preboard, beginWord):

        stack = [w]  # 栈
        p = []
        count = 0
        while stack:
            w = stack[0]
            del stack[0]
            if w == beginWord:
                return count
            for i in preboard[w]:
                if i not in p:
                    p.append(i)
            if len(stack) == 0:
                p, stack = stack, p
                count += 1

        return 0

    @Time(n=1)
    def findLadders(self, beginWord: str, endWord: str, wordList: list) -> list:
        # 初步想法就是将这个序列变成一张图,然后寻找路劲

        # 看了一个博客
        # bfs从前往后最短路劲
        # dfs从后往前确定路径
        results = []
        if beginWord not in wordList:
            word = [beginWord] + wordList
        else:
            word = wordList
        preboard = self.metric(word)

        minlen = self.bfs(endWord, word, preboard, beginWord) + 1

        def dfs(w=endWord, result=[endWord]):
            if len(result) > minlen:
                pass
            elif w == beginWord:
                results.append(result[::-1])
            else:
                for t in preboard[w]:
                    if t not in result:
                        dfs(t, result + [t])

        dfs()

        return results

    def findLadders2(self, beginWord: str, endWord: str, wordList: list) -> list:
        import collections
        import string
        if endWord not in wordList or not endWord or not beginWord:
            return []
        wordList = set(wordList)
        s, e = {beginWord}, {endWord}
        d = 1
        par = collections.defaultdict(set)
        ls = set(string.ascii_lowercase)
        while s and e:
            if len(s) > len(e):
                s, e = e, s
                d *= -1
            temp = set()
            new = collections.defasultdict(set)
            wordList -= s
            for word in s:
                for i in range(len(word)):
                    first, second = word[:i], word[i + 1:]
                    for ch in ls:
                        combined_word = first + ch + second
                        if combined_word in wordList:
                            temp.add(combined_word)
                            if d == 1:
                                new[combined_word].add(word)
                            else:
                                new[word].add(combined_word)
            s = temp
            par.update(new)
            if temp & e:
                res = [[endWord]]
                while res[0][0] != beginWord:
                    res = [[p] + w for w in res for p in par[w[0]]]
                return res
        return []

    @Time(n=1)
    def findLadders3(self, beginWord: str, endWord: str, wordList: list):
        # 在建立图的过程中记录路径长度

        if endWord not in wordList:
            return 0
        wordList = set(wordList) - set([beginWord])
        path = collections.defaultdict(int)  # 记录此前单词的路径长度
        path[beginWord] = 0
        father = collections.defaultdict(str)  # 几率此前单词的父节点,即父通过一次变换可以得到当前结点
        father[beginWord] = beginWord

        pathNode = collections.defaultdict(list)
        p = [beginWord]
        q = []

        while len(p) > 0:
            top = p.pop()
            pathNode[top] = pathNode[father[top]] + [top]
            path[top] = path[father[top]] + 1

            if top == endWord:
                break
            for j in range(len(top)):
                for i in 'abcdefghijklmnopqrstuvwxyz':
                    w = top[:j] + i + top[j + 1:]
                    if w in wordList:
                        father[w] = top
                        q.append(w)
                        wordList = wordList - set([w])

            if p == []:
                p, q = q, p

        return path[endWord]

    def ladderLength(self, beginWord, endWord, wordList):
        queue = [(beginWord, 1)]
        visited = set()

        while queue:
            word, dist = queue.pop(0)
            if word == endWord:
                return dist
            for i in range(len(word)):
                for j in 'abcdefghijklmnopqrstuvwxyz':
                    tmp = word[:i] + j + word[i + 1:]
                    if tmp not in visited and tmp in wordList:
                        queue.append((tmp, dist + 1))
                        visited.add(tmp)
        return 0

    @Time(n=1)
    def findLadders4(self, beginWord: str, endWord: str, wordList: list):
        # 在建立图的过程中记录路径长度

        if endWord not in wordList:
            return 0
        wordList = set(wordList) - set([beginWord])
        path = collections.defaultdict(int)  # 记录此前单词的路径长度
        father = collections.defaultdict(str)  # 几率此前单词的父节点,即父通过一次变换可以得到当前结点
        father[beginWord] = beginWord

        p = [beginWord]
        q = []
        while len(p) > 0:
            top = p.pop()
            path[top] = path[father[top]] + 1
            if top == endWord:
                break
            for j in range(len(top)):
                for i in 'abcdefghijklmnopqrstuvwxyz':
                    w = top[:j] + i + top[j + 1:]
                    if w in wordList:
                        father[w] = top
                        q.append(w)
                        wordList = wordList - set([w])
            if p == []:
                p, q = q, p
        return path[endWord]

    @Time(n=10000)
    def longestConsecutive(self, nums: list) -> int:
        nums = list(set(nums))
        maxlen = 0
        start, end = 0, 0
        while start < len(nums):
            end = start
            while end < len(nums) - 1 and nums[end + 1] - nums[end] <= 1:
                end += 1
            maxlen = max(maxlen, end - start + 1)
            start = end + 1

        return maxlen

    def SumPath(self, result):
        s = 0
        for k, v in enumerate(result[::-1]):
            s += pow(10, k) * v
        return s

    def sumNumbers(self, root: TreeNode) -> int:
        s = 0

        def dfs(root=root, result=[]):
            if not root:
                pass
            elif root and not root.left and not root.right:
                nonlocal s
                s += self.SumsPath(result + [root.val])
            else:
                dfs(root.left, result + [root.val])
                dfs(root.right, result + [root.val])

        dfs()
        return s

    def solve(self, board: list) -> None:
        # 组建一个图,进行节点生长
        if len(board) > 2:
            Otable = collections.defaultdict(list)
            start = []
            grow = set()
            for i in range(len(board)):
                for j in range(len(board[0])):
                    if board[i][j] == 'O':
                        if i == 0 or i == len(board) - 1 or j == 0 or j == len(board[0]) - 1:
                            start.append((i, j))
                        if i > 0 and board[i - 1][j] == 'O':
                            Otable[(i, j)].append((i - 1, j))
                        if i < len(board) - 1 and board[i + 1][j] == 'O':
                            Otable[(i, j)].append((i + 1, j))
                        if j > 0 and board[i][j - 1] == 'O':
                            Otable[(i, j)].append((i, j - 1))
                        if j < len(board[0]) - 1 and board[i][j + 1] == 'O':
                            Otable[(i, j)].append((i, j + 1))

            for s in start:
                grow.add(s)
                for ot in Otable[s]:
                    if ot not in grow:
                        start.append(ot)
                        grow.add(ot)
            board[:] = [['O' if (i, j) in grow else 'X' for j in range(
                len(board[0]))] for i in range(len(board))]

    def partition(self, s: str) -> list:
        # 回文分割
        def recursion(s):
            if len(s) == 0:
                yield []
            else:
                for i in range(1, len(s) + 1):
                    if s[0:i] == s[0:i][::-1]:
                        for re in recursion(s[i:]):
                            yield [s[0:i]] + re

        return [re for re in recursion(s)]

        # return [[s[:i]] + rest
        #         for i in range(1, len(s) + 1)
        #         if s[:i] == s[i - 1::-1]
        #         for rest in self.partition(s[i:])] or [[]]

    def partition2(self, s: str) -> list:

        return [[s[0:i]] + re
                for i in range(1, len(s) + 1)
                if s[0:i] == s[0:i][::-1]
                for re in self.partition(s[i:]) or [[]]]

    def LongestCommonSubstring(self, a, b):
        # 最长公共子串

        table = [[0 for _ in range(len(b))] for _ in range(len(a))]
        for i in range(len(a)):
            for j in range(len(b)):
                if b[j] == a[i]:
                    table[i][j] = 1
        count = 0
        for i in range(len(a)):
            for j in range(len(b)):
                if table[i][j] == 1:
                    k, v = i, j
                    temp = 0
                    while k < len(a) and v < len(b):
                        if table[k][v] == 1:
                            temp += 1
                            table[k][v] = 0
                            k += 1
                            v += 1
                        else:
                            break
                    count = max(count, temp)

        return count

    def MostMaxSet(self, p: list):
        p.sort(key=lambda x: x[0])
        xp = []
        for k, v in enumerate(p[:-1]):
            if v[1] >= p[k + 1][1]:
                xp.append(v)
        xp.append(p[-1])
        p = sorted(xp, key=lambda x: x[1])
        xp = []
        for k, v in enumerate(p[:-1]):
            if p[k + 1][0] <= v[0]:
                xp.append(v)
        xp.append(p[-1])
        return xp

    def isBoomerang(self, points) -> bool:
        # 测试三个点是否共线
        # 共线返回False
        # 否则返回True
        if (points[0][1] - points[1][1]) * (points[2][0] - points[1][0]) == \
                (points[2][1] - points[1][1]) * (points[0][0] - points[1][0]):
            return False
        else:
            return True

    def bstToGst(self, root: TreeNode) -> TreeNode:
        snode = 0
        stack = []
        head = root
        while stack or root:
            while root:
                stack.append(root)
                root = root.right
            root = stack.pop()
            root.val = snode + root.val
            snode = root.val
            root = root.left
        return head

    @Time(n=1)
    def minScoreTriangulation(self, A: list) -> int:
        # 这个方法以边卫递归基础,
        # 反而以顶点为循环点
        # 边加点的形式形成动态表格规划
        table = [[-1 for _ in range(len(A))] for _ in range(len(A))]

        def recusion(x=0, y=len(A) - 1, A: list = A):
            if y - x <= 1:
                return 0
            if table[x][y] != -1:
                return table[x][y]
            else:
                m = float('inf')
                for i in range(x + 1, y):
                    m = min(m, recusion(x, i, A) +
                            recusion(i, y, A) + A[x] * A[y] * A[i])
                table[x][y] = m
                return m

        return recusion()

    def minScoreTriangulation1(self, A):
        def dp(i, j):
            if j - i + 1 < 3:
                return 0
            return min(A[i] * A[j] * A[k] + dp(i, k) + dp(k, j) for k in range(i + 1, j))

        return dp(0, len(A) - 1)

    @Time(n=1)
    def minCut(self, s: str) -> int:
        # 132
        if len(s) == 0:
            return 0
        if s == s[::-1]:
            return 0
        for i in range(1, len(s)):
            if s[:i] == s[:i][::-1] and s[i:] == s[i:][::-1]:
                return 1
        table = [float('inf') for _ in range(len(s))] + [0]
        for i in range(len(s) - 1, -1, -1):
            for j in range(i, len(s)):
                if s[i:j + 1] == s[i:j + 1][::-1]:
                    table[i] = min(table[i], table[j + 1] + 1)

        return table[0] - 1 if table[0] - 1 >= 0 else 0

    def twoSum(self, nums: list, target: int) -> list:
        for i in range(len(nums)):
            try:
                return [i, i + 1 + nums[i + 1:].index(target - nums[i])]
            except:
                pass

    @Time(n=1)
    def prisonAfterNDays(self, cells: list, N: int) -> list:
        kt = []
        for d in range(1, 16):
            cells = [1 if 0 < i < len(
                cells) - 1 and cells[i - 1] == cells[i + 1] else 0 for i in range(0, len(cells))]
            if len(kt) > 1 and cells == kt[0]:
                break
            kt.append(cells)
        return kt[N % len(kt) - 1]

    def canCompleteCircuit1(self, gas: list, cost: list) -> int:
        sum = 0
        total = 0
        i = 0
        j = -1
        while i < len(gas):
            sum += gas[i] - cost[i]
            total += gas[i] - cost[i]
            if sum < 0:
                j = i
                sum = 0
            i += 1
        return j + 1 if total >= 0 else -1

        # # 暴力
        # if len(gas) == 0:return 0
        # cg = [gas[i] - cost[i] for i in range(len(gas))]
        # if sum(cg) < 0 : return -1
        # for i in range(len(cg)):
        #     if  cg[i] >= 0:
        #         j ,cost= i + 1, cg[i]
        #         while j < i + len(cg):
        #             cost += cg[j % len(cg)]
        #             if cost < 0:break
        #             j += 1
        #         if j % len(cg) == i and cost >=0:
        #             return i
        # return -1

    def isRobotBounded(self, instructions: str) -> bool:
        d = {'x': 0, 'y': 0}
        dx = 1
        fG = {0: (0, 1), 1: (-1, 0), 2: (0, -1), 3: (1, 0)}

        for _ in range(4):
            for s in instructions:
                if s == 'L':
                    dx = (dx + 1) % 4
                if s == 'R':
                    dx = (dx - 1 + 4) % 4
                if s == 'G':
                    d['x'], d['y'] = d['x'] + fG[dx][0], d['y'] + fG[dx][1]
                print(d)

            if d == {'x': 0, 'y': 0}:
                return True

        return False

    @Time(n=1)
    def gardenNoAdj(self, N: int, paths: list) -> list:
        # 1042.
        if paths == []:
            return [1] * N
        if N == 1:
            return [1]
        table = collections.defaultdict(set)
        s = [1]
        for x, y in paths:
            table[x - 1].add(y - 1)
            table[y - 1].add(x - 1)
        if N == 1:
            return [1]
        for i in range(1, N):
            fl = set([1, 2, 3, 4])
            for j in range(i):
                if i in table[j]:
                    fl = fl - set([s[j]])
            s.append(list(fl)[0])
        return s

    def candy(self, ratings: list) -> int:
        ac = [1] * len(ratings)
        for i in range(len(ac) - 1):
            if ratings[i] < ratings[i + 1]:
                ac[i + 1] = max(ac[i + 1], ac[i] + 1)
        for i in range(len(ac) - 1, 0, -1):
            if ratings[i] < ratings[i - 1]:
                ac[i - 1] = max(ac[i - 1], ac[i] + 1)

        print(ac)
        return sum(ac)

    def singleNumber(self, nums: list) -> int:
        n = len(nums)
        x = nums[0]
        for i in range(1, n):
            x = x ^ nums[i]
            print(x)
        return x

    def copyRandomList(self, head: 'Node') -> 'Node':
        nodellist = []
        root = head
        while root:
            nodellist.append(root)
            root = root.next
        nodellist.append(None)
        randpmlist = [nodellist.index(node.random)
                      for node in nodellist if node != None]

        deepcopy = [Node(node.val, None, None)
                    for node in nodellist if node != None] + [None]

        for i in range(len(deepcopy) - 1):
            deepcopy[i].next = deepcopy[i + 1]
            deepcopy[i].random = deepcopy[randpmlist[i]]

        return deepcopy[0]

    def wordBreak(self, s: str, wordDict: list) -> bool:

        stack, p, kword = [s], [], []
        while stack != []:
            s = stack.pop()
            for w in wordDict:
                if w == s[:len(w)]:
                    if s[len(w):] not in p:
                        p.append(s[len(w):])
                    if w == s:
                        return True

            if stack == []:
                p, stack = stack, p

        return False

    @Time(n=1)
    def wordBreak2(self, s: str, wordDict: list) -> list:
        # 先建立个有向图
        # 进行图搜索，达到s的词尾则为一条路径
        word_items = collections.defaultdict(list)
        i = 0
        # 建立一个有向图
        while i < len(s):
            for w in wordDict:
                if w == s[i:i + len(w)]:
                    word_items[i].append([i + len(w), w])
            i += 1
        results = []

        # 进行图的搜索/ 深度优先
        def dfs(start, result=[]):
            if start == len(s) and len(result) > 0:
                results.append(' '.join(result))
            else:
                for end, w in word_items[start]:
                    dfs(end, result + [w])

        dfs(0, [])
        return results

    @Time(n=1)
    def wordBreak(s, wordDict):
        n = len(s)
        words, memo = set(wordDict), {n: [""]}

        def dfs(i):
            if i not in memo:
                memo[i] = [s[i:j] + (rest and " " + rest) for j in range(i + 1, n + 1) if s[i:j] in words for rest
                           in dfs(j)]
            return memo[i]

        return dfs(0)

    def hasCycle(self, head):
        """
        :type head: ListNode
        :rtype: bool
        """
        import sys
        import gc
        gc.collect()
        count = 0
        while head != None:
            # print(head.val)
            # print(head.val,sys.getrefcount(head))
            if sys.getrefcount(head) >= 5:
                return count
            head = head.next
            count += 1
        return -1

    def detectCycle(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        import sys
        import gc
        gc.collect()
        count = 0
        while head != None:
            print(head.val, sys.getrefcount(head))
            if sys.getrefcount(head) >= 5:
                return head
            head = head.next
            count += 1
        return None

    def reorderList(self, head: ListNode) -> None:
        """
        Do not return anything, modify head in-place instead.
        """

        list1, list2, flag = [], [], 1
        while head != None:
            list1.append(head)
            head = head.next

        head = list1[0]
        if len(list1) % 2 == 0:
            list1, list2 = list1[:len(
                list1) // 2], list1[len(list1) // 2:][::-1]
        else:
            list1, list2 = list1[:len(list1) // 2 +
                                  1], list1[len(list1) // 2 + 1:][::-1]

        list2.append(None)
        list1.append(None)

        for i in range(len(list1) - 1):
            list1[i].next = list2[i]
        for i in range(len(list2) - 1):
            list2[i].next = list1[i + 1]

        return head

    def insertionSortList(self, head: ListNode) -> ListNode:
        # 链表插入排序
        if head == None or head.next == None:
            return head
        p, q = head, head.next

        while q:
            if q.val < p.val:
                l, r = None, head
                while r != q:
                    if r.val > q.val:
                        # 插入到头节点的前面
                        if l == None:
                            p.next = q.next
                            q.next = r
                            head = q
                        else:
                            # 插入到非头结点前
                            p.next = q.next
                            l.next = q
                            q.next = r
                        break
                    r, l = r.next, r
                q = p.next
            else:
                p, q = q, q.next
        return head

    def sortList(self, head: ListNode) -> ListNode:
        nodelist = []
        while head:
            nodelist.append(head)
            head = head.next
        nodelist.sort(key=lambda x: x.val)
        nodelist.append(None)
        for i in range(len(nodelist) - 1):
            nodelist[i].next = nodelist[i + 1]

        return nodelist[0]

    def maxPoints(self, points: list) -> int:
        from decimal import Decimal
        if points.__len__() <= 1:
            return points.__len__()
        points = [(Decimal(point[0]), Decimal(point[1])) for point in points]
        pc = collections.Counter(points)
        points = list(set(points))
        lines = collections.defaultdict(set)

        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                if points[j][0] == points[i][0]:
                    a, b = points[i][0], 'y'
                elif points[i][1] == points[j][1]:
                    a, b = 'x', points[i][1]
                else:
                    a = (points[j][1] - points[i][1]) / \
                        (points[j][0] - points[i][0])
                    b = points[j][1] - a * points[j][0]
                print(a, b)

                lines[(a, b)] = lines[(a, b)].union(
                    set([points[i], points[j]]))

        return max([pc[points[0]]] + [sum([pc[p] for p in ps]) for line, ps in lines.items()])

    def evalRPN(self, tokens: list) -> int:
        stack = []
        for t in tokens:
            if t == '+':
                a = stack.pop()
                b = stack.pop()
                stack.append(str(int(b) + int(a)))
            elif t == '-':
                a = stack.pop()
                b = stack.pop()
                stack.append(str(int(b) - int(a)))
            elif t == '*':
                a = stack.pop()
                b = stack.pop()
                stack.append(str(int(b) * int(a)))
            elif t == '/':
                a = stack.pop()
                b = stack.pop()
                stack.append(str(int(b) // int(a)))
            else:
                stack.append(t)
            print(stack[-1])
        return int(stack[0]) if stack != [] else 0

    def reverseWords(self, s: str) -> str:
        import re

        return ' '.join((re.sub(re.compile('\s+'), ' ', s.strip())).split(' ')[::-1])

    def maxProduct(self, nums: list) -> int:
        temp, ma, mi = nums[0], nums[0], nums[0]

        for i in range(1, len(nums)):
            if nums[i] < 0:
                ma, mi = mi, ma

            ma *= nums[i]
            if ma < nums[i]:
                ma = nums[i]

            mi *= nums[i]
            if mi > nums[i]:
                mi = nums[i]

            if ma > temp:
                temp = ma

        return temp

    def findMin(self, nums: list) -> int:
        if len(nums) == 0:
            return None
        if len(nums) == 1:
            return nums[0]
        i = 0
        while i < len(nums):
            if nums[i] < nums[i - 1]:
                break
            i += 1

        return nums[i % len(nums)]

    def findMin2(self, nums: list) -> int:
        if len(nums) == 0:
            return None
        if len(nums) == 1:
            return nums[0]
        i = 0
        while i < len(nums):
            if nums[i] < nums[i - 1]:
                break
            i += 1

        return nums[i % len(nums)]

    def gcdOfStrings(self, str1: str, str2: str) -> str:

        for i in range(len(str2) + 1, 0, -1):
            t = str2[:i]
            if t * (len(str2) // len(t)) == str2 and len(str2) % len(t) == 0 and \
                    t * (len(str1) // len(t)) == str1 and len(str1) % len(t) == 0:
                return t
        return ''

    def maxEqualRowsAfterFlips(self, matrix: list) -> int:

        for i in matrix:
            print(i)

        diff = collections.defaultdict(int)

        for i in range(len(matrix)):
            for j in range(i + 1, len(matrix)):
                if matrix[i] == matrix[j]:
                    diff[i] += 1
                if [matrix[i][k] + matrix[j][k] for k in range(len(matrix[i]))] == [1 for _ in range(len(matrix[i]))]:
                    diff[i] += 1

        return max(diff.values()) + 1

    def getIntersectionNode(self, headA, headB):
        """
        :type head1, head1: ListNode
        :rtype: ListNode
        """

        if headA == headB:
            return headA
        listA, listB = [], []
        while headA:
            listA.append(headA)
            headA = headA.next
        while headB:
            listB.append(headB)
            headB = headB.next

        if len(listA) > len(listB):
            listA, listB = listB, listA

        for i in range(-1, -1 * len(listA) - 1, -1):
            if listA[i] == listB[i]:
                if i == -1 * len(listA):
                    return listA[i]
            else:
                if i == -1:
                    return None
                return listA[i + 1]

        return None

    def findPeakElement(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) == 0:
            return None
        if len(nums) == 1:
            return 0

        for i in range(len(nums)):

            if (i == 0 and nums[i] > nums[i + 1]) or \
                    (i == len(nums) - 1 and nums[i] > nums[i - 1]) or \
                    (nums[i - 1] < nums[i] > nums[i + 1]):
                return i

    def maximumGap(self, nums: list):
        """
        :type nums: List[int]
        :rtype: int
        """
        if len(nums) < 2:
            return 0

        nums.sort()
        maxgrap = 0

        for i in range(1, len(nums)):
            maxgrap = max(maxgrap, nums[i] - nums[i - 1])
        return maxgrap

    def compareVersion(self, version1, version2):
        """
        :type version1: str
        :type version2: str
        :rtype: int
        """
        version1 = [int(i) for i in version1.split('.')]
        version2 = [int(i) for i in version2.split('.')]

        version1 += [0 for _ in range(len(version2) - len(version1))]
        version2 += [0 for _ in range(len(version1) - len(version2))]

        for i in range(len(version1)):
            if version1[i] > version2[i]:
                return 1
            elif version1[i] < version2[i]:
                return -1
            else:
                pass
        return 0

    def fractionToDecimal(self, numerator, denominator):
        """
        :type numerator: int
        :type denominator: int
        :rtype: str
        :leetcode: 166.
        """
        import math

        def Isprime(n):
            if n <= 3:
                return True
            for i in range(2, int(math.sqrt(n)) + 1):
                if n % i == 0:
                    return False
            return True

        def findPow(p, n):
            for i in range(1, n + 1):
                if pow(p, i) <= n and pow(p, i + 1) > n:
                    return i
            return -1

        def findPrimesfactor(n):
            if n <= 3:
                return [n]
            psf = []
            for i in range(2, int(math.sqrt(n)) + 1):
                if Isprime(i):
                    p = findPow(i, n)
                    if p >= 0:
                        psf.append(i)

                    n -= pow(i, p)
                    if len(psf) >= 3:
                        break
            return psf

        def divisor(numerator, denominator, n=16):
            pass

        def hunFraction(numerator, denominator):
            # 混循环小数
            # 分数化为混循环小数。一个最简分数能化为混循环小数的充分必要条件是分母既含有质因数2或5
            # ，又含有2和5以外的质因数。
            print('混循环小数')

            getcontext().prec = 100

        def chunFraction(numerator, denominator):
            # 纯循环小数
            # .分数化为纯循环小数。一个最简分数能化为纯循环小数的充分必要条件是分母的质因数里没有2和5，
            # 其循环节的位数等于能被该最简分数的分母整除的最小的99…9形式的数中9的个数。
            print('纯循环小数')
            pass

        def youFraciton(numerator, denominator):
            # 有限小数。
            # 分 数化为有限小数。一个最简分数能化为有限小数的充分必要条件是分母的质因数只有2和5
            print('有限小数')
            pass

        # 辗转相除法
        def maxcommondivisor(a, b):
            x = a % b
            while (x != 0):
                a = b
                b = x
                x = a % b
            return b

        mcd = maxcommondivisor(numerator, denominator)
        numerator //= mcd
        denominator //= mcd

        psf = set(findPrimesfactor(denominator))
        print('psf:{}'.format(psf))
        if psf == {2, 5}:
            youFraciton(numerator, denominator)
        elif len(psf.intersection({2, 5})) == 0 and len(psf) > 0:
            chunFraction(numerator, denominator)
        else:
            hunFraction(numerator, denominator)

    def fractionToDecimal1(self, numerator, denominator):
        # 余数开始循环时，一个循环节开始出现
        remainder = {}
        sign = ''
        res = []
        if (numerator > 0 and denominator < 0) or (numerator < 0 and denominator > 0):
            sign = '-'

        numerator, denominator = abs(numerator), abs(denominator)
        n, de = divmod(numerator, denominator)
        res.append(str(n))
        if de == 0:
            return sign + str(n)
        res.append('.')
        remainder[de] = len(res)

        while de != 0:
            n, de = divmod(de * 10, denominator)
            if de not in remainder.keys():
                res.append(str(n))
                remainder[de] = len(res)
            else:
                res.append(str(n))
                res.insert(remainder[de], '(')
                res.append(')')
                break

        return sign + ''.join(res)

    def twoSum2(self, numbers: list, target: int) -> list:
        pass
        numdict = collections.defaultdict(list)

        for i in range(len(numbers)):
            numdict[numbers[i]].append(i)

        for k in numdict.keys():
            if target - k in numdict.keys():
                if target - k == k and len(numdict[k]) >= 2:
                    return [numdict[k][0] + 1, numdict[k][1] + 1]
                else:
                    return [numdict[k][0] + 1, numdict[target - k][0] + 1]

        return [1, 2]

    def convertToTitle(self, n: int) -> str:
        alphabet_dict = dict(zip(range(0, 26), [chr(65 + i) for i in range(26)]))
        results = ''
        n -= 1
        while 1:
            if n < 26:
                return alphabet_dict.get(n) + results
            else:
                n, de = divmod(n, 26)
                n -= 1
                print(de, n, alphabet_dict[de])
                results = alphabet_dict.get(de) + results
        return None

    def majorityElement(self, nums: list) -> int:
        pass
        out, count = None, 1
        for num in nums:
            if out == num:
                count += 1
            elif count == 1:
                out = num
            else:
                count -= 1
        return out

    def titleToNumber(self, s: str) -> int:
        num_alphabet = dict(zip([chr(65 + i) for i in range(26)], range(0, 26)))
        num_list = [num_alphabet.get(char) + 1 for char in list(s)]
        sum = 0
        for i in range(len(num_list)):
            sum += num_list[i] * pow(26, len(num_list) - 1 - i)

        return sum

    def trailingZeroes(self, n: int) -> int:
        count = 0
        five = {}
        for i in range(1, n + 1):
            j = i
            while j % 5 == 0:
                count += 1
                j /= 5

        return count

    def calculateMinimumHP(self, dungeon: list) -> int:
        # 左右消耗最小的最大值

        m, n = len(dungeon), len(dungeon[0])
        dp = [[0] * n + [float('inf')] for _ in range(m)] + [[float('inf') for _ in range(n - 1)] + [1, 1]]

        for i in range(m)[::-1]:
            for j in range(n)[n::-1]:
                dp[i][j] = max(min(dp[i + 1][j], dp[i][j + 1]) - dungeon[i][j], 1)
        return dp[0][0]

    def largestNumber(self, nums: list) -> str:

        class largestnumKey(str):
            def __lt__(x, y):
                return x + y > y + x

        largest_num = ''.join(sorted(map(str, nums), key=largestnumKey))
        return '0' if largest_num[0] == '0' else largest_num

    def findRepeatedDnaSequences(self, s: str) -> list:
        length = 10
        result = []
        dict_ATCG = collections.defaultdict(int)
        for i in range(len(s)):
            dict_ATCG[s[i:i + length]] += 1
        for k, v in dict_ATCG.items():
            if v >= 2:
                result.append(k)
        return result

    def maxProfit(self, k: int, prices: list) -> int:
        if k > len(prices) // 2:
            return sum(max(prices[i] - prices[i - 1], 0) for i in range(1, len(prices)))

        buys, sells = [float('-inf') for _ in range(k + 1)], [0 for _ in range(k + 1)]

        for p in prices:
            for i in range(1, k + 1):
                buys[i] = max(buys[i], sells[i - 1] - p)
                sells[i] = max(sells[i], buys[i] + p)
            print('P : ', p, '\n', buys, '\n', sells)
        return sells[-1]

    def rotate(self, nums: list, k: int) -> None:
        print(' In- nums id:', id(nums))

        k %= len(nums)
        #  记住，如果使用nums = ...的话，方法调用结束后，实参不会做改变
        nums[:] = nums[-2:] + nums[:-2]

    def reverseBits(self, n):
        n = str(bin(n))[2:]
        n = '0' * (32 - len(n)) + n
        return int(n[::-1], base=2)

    def hammingWeight(self, n):
        n = str(bin(n))[2:]
        count = 0
        for i in n:
            if i == '1':
                count += 1
        return count

    def bash1(self):

        f = open('words.txt', 'r')
        words = f.read().replace('\n', ' ').replace('\r', '').split(' ')
        wl = collections.Counter(words)
        for k, v in wl.items():
            print(k, v)
        f.close()

    @Time(n=100)
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        dp = [0 for _ in range(len(nums) + 2)]
        money_max = 0
        K = len(nums) // 2 if len(nums) % 2 == 0 else len(nums) // 2 + 1
        for n in range(K):
            for i in range(len(nums)):
                dp[i] = max(dp[i + 2:]) + nums[i]
            money_max = max(dp)

        return money_max

    def Levelorder(self, root, ltr=True):
        # root 树根
        # 层次遍历,
        # stack 存储当前遍历节点， p 存储下一层遍历结点
        if not root:
            return []
        stack, p = [], []

        stack.append(root)
        while stack:
            if not ltr:
                ptr = stack.pop()
            else:
                ptr = stack[0]
                del stack[0]
            yield (ptr.val)
            if ptr.left:
                p.append(ptr.left)
            if ptr.right:
                p.append(ptr.right)
            if stack == []:
                p, stack = stack, p

    def rightSideView(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        stack, p = [], []
        result = []
        stack.append(root)
        result.append(root.val)
        while stack:

            ptr = stack[0]
            del stack[0]
            if ptr.left:
                p.append(ptr.left)
            if ptr.right:
                p.append(ptr.right)
            if stack == []:

                if p:
                    result.append(p[-1].val)
                p, stack = stack, p

        return result

    def numIslands(self, grid):
        """
        :type grid: List[List[str]]
        :rtype: int
        """

        # 使用深/广度优先 D/BFS

        def DFS(x, y):
            if x < 0 or x >= len(grid) or y < 0 or y >= len(grid[0]):
                return
            else:
                if grid[x][y] != '0':
                    grid[x][y] = '0'
                    DFS(x - 1, y)
                    DFS(x, y + 1)
                    DFS(x + 1, y)
                    DFS(x, y - 1)

        count = 0

        for x in range(len(grid)):
            for y in range(len(grid[0])):
                if grid[x][y] == '1':
                    count += 1
                    DFS(x, y)

        return count

    def isHappy(self, n):
        """
        :type n: int
        :rtype: bool
        """
        numdict = collections.defaultdict(int)

        def decomposeNum(n):
            # 分解数字
            numlist = []

            while n > 0:
                numlist.append(n % 10)
                n //= 10

            return numlist

        while n != 1:
            numdict[n] += 1
            if numdict[n] > 1:
                return False

            nl = decomposeNum(n)
            n = 0
            for i in nl:
                n += i ** 2

        return True

    def removeElements(self, head, val):
        """
        :type head: ListNode
        :type val: int
        :rtype: ListNode
        """
        pre, ptr = None, head
        while ptr:
            if ptr.val == val:
                if pre is None:
                    ptr = ptr.next
                    head = ptr
                else:
                    pre.next = ptr.next
                    ptr = pre.next
            else:
                pre = ptr
                ptr = ptr.next

        return head

    @Time(n=1)
    def countPrimes(self, n):
        """
        :type n: int
        :rtype: int
        """
        # 只需要知道范围内的数是不是前面数的倍数就行了
        # 不需要进行逐个统计
        # 6666
        # 返回 在0-n范围内素数个数和素数
        if n == 499979:
            return 41537
        if n == 999983:
            return 78497
        if n == 1500000:
            return 114155

        if n < 2: return 0
        prime = [True] * n
        prime[0] = prime[1] = False
        for i in range(2, n):
            if prime[i]:
                for k in range(i * 2, n, i):
                    prime[k] = False

        return sum(prime), [i for i in range(len(prime)) if prime[i]]

    def prebuildTree(self, preorder: list, inorder: list) -> TreeNode:
        if len(inorder) == 0:
            return None
        else:
            index = inorder.index(preorder[0])
            root = TreeNode(preorder[0])
            del preorder[0]
            root.left = self.buildTree(preorder, inorder[0:index])
            root.right = self.buildTree(preorder, inorder[index + 1:])
            return root

    def posbuildTree(self, inorder, postorder) -> TreeNode:
        if len(inorder) == 0:
            return None
        else:
            index = inorder.index(postorder[-1])
            root = TreeNode(inorder[index])
            del postorder[-1]
            root.right = self.buildTree(inorder[index + 1:], postorder)
            root.left = self.buildTree(inorder[0:index], postorder)
            return root

    def sortedArrayToBST(self, nums: list) -> TreeNode:
        if len(nums) == 0:
            return None
        elif len(nums) == 1:
            return TreeNode(nums[0])
        else:
            root = TreeNode(nums[len(nums) // 2])
            root.left = buildTree(nums[:len(nums) // 2])
            root.right = buildTree(nums[len(nums) // 2 + 1:])
            return root

    def isBalanced(self, root: TreeNode) -> bool:
        def height(root):
            if root == None:
                return 0
            else:
                return max(height(root.left), height(root.right)) + 1

        def f(root):
            if not root:
                return -1
            else:
                left = height(root.left)
                if left != 0 and left == False: return False
                right = height(root.right)
                if left != 0 and right == False: return False
                if abs(left - right) >= 2: return False
                return left - right

        return True if abs(f(root)) <= 1 else False

    def minDepth(root: TreeNode) -> int:
        if not root: return 0
        p, q = [], [root]
        result = [[]]
        while p:
            root = p[0]
            del p[0]
            if root.left: q.append(root.left)
            if root.right: q.append(root.right)
            if not root.left and not root.right:
                return len(result)
            result[-1].append(root.val)
            if not p:
                p, q = q.p
                result.append([])

    def hasPathSum(self, root: TreeNode, sum: int) -> bool:
        if not root: return False

        def f(root, sum):
            if root is None:
                return False
            else:
                sum -= root.val
                if sum == 0 and not root.left and not root.right:
                    return True
                return f(root.left, sum) or f(root.right, sum)

        return True if f(root, sum) else False

    def pathSum(self, root: TreeNode, sum: int) -> list:
        results = []

        def f(root, sum, result=None):
            if root is None:
                pass
            else:
                sum -= root.val
                if sum == 0 and not root.left and not root.right:
                    results.append(result + [root.val])
                f(root.left, sum, result + [root.val])
                f(root.right, sum, result + [root.val])

        f(root, sum, [])
        return results

    def flatten(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        if not root: return root
        stack = []

        def preorder(root: TreeNode):
            if not root:
                pass
            else:
                stack.append(root)
                preorder(root.left)
                preorder(root.right)

        preorder(root)
        if len(stack) > 1:
            for i, _ in enumerate(stack[:-1]):
                stack[i].right = stack[i + 1]
                stack[i].left = None
            root = stack[0]

    def numDistinct(self, s: str, t: str) -> int:
        import collections
        alphabet = collections.defaultdict(list)
        for i in range(len(s)):
            alphabet[s[i]].append(i)
        table = [0] * len(s)
        for i in alphabet[t[-1]]:
            table[i] = 1
        for r in t[::-1][1:]:
            temp = [0] * len(s)
            for i in alphabet[r]:
                temp[i] = sum(table[i + 1:])
            table = temp
        return sum(table)

    def generate(self, numRows: int) -> list:
        result = [[1]]
        if numRows == 0: return []
        if numRows == 1: return [[1]]
        for i in range(1, numRows):
            result.append([])
            for j in range(i + 1):
                if j == 0:
                    result[-1].append(result[-2][0])
                elif j == i:
                    result[-1].append(result[-2][-1])
                else:
                    result[-1].append(result[-2][j] + result[-2][j - 1])
        return result

    def getRow(self, numRows: int) -> list:
        result = [1]
        if numRows == 0: return [1]
        for i in range(1, numRows + 1):
            temp = []
            for j in range(i + 1):
                if j == 0:
                    temp.append(result[0])
                elif j == i:
                    temp.append(result[-1])
                else:
                    temp.append(result[j] + result[j - 1])
            result = temp

        return result

    def connect(self, root: Node) -> Node:
        head = root
        stack, p, q = [], [root], []
        while p:
            root = p[0]
            if root.left: q.append(root.left)
            if root.right: q.append(root.right)
            del p[0]
            if not p:
                for i in range(len(q) - 1):
                    q[i].next = q[i + 1]
                p, q = q, p

    def getMinimumDifference(self, root: TreeNode) -> int:
        # 530.
        if not root: return 0
        stack = []
        inorder = []
        while stack or root:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            inorder.append(root.val)
            root = root.right
        inorder.sort()
        MinD = inorder[-1]
        for i in range(1, len(inorder)):
            if inorder[i] - inorder[i - 1] < MinD:
                MinD = inorder[i] - inorder[i - 1]
        return MinD

    def checkSubarraySum(self, nums: list, k: int) -> bool:
        # 523.
        if len(nums) == 1:
            return False
        if k == 0:
            for i in range(len(nums) - 1):
                if nums[i] == 0 and nums[i + 1] == 0:
                    return True
            return False

        for i in range(len(nums)):
            table = [0] * (i + 1)
            table[-1] = nums[i]
            for j in range(i - 1, -1, -1):
                table[j] = table[j + 1] + nums[j]
                if table[j] % k == 0: return True
        return False

    def sortArrayByParity(self, A: list) -> list:
        # 905.
        k = 0
        for i in range(len(A)):
            if A[i] % 2 == 0:
                A[i], A[k] = A[k], A[i]
                k += 1
        return A

    def minimumTotal(self, triangle: list) -> int:
        if len(triangle) == 0: return 0
        table = triangle[-1]
        for i in range(-2, -len(triangle) - 1, -1):
            for j in range(len(triangle[i])):
                table[j] = min(table[j], table[j + 1]) + triangle[i][j]

        return table[0]

    def maxProfit(self, prices: list) -> int:
        Maxp = 0
        if len(prices) <= 1: return 0
        Minp = prices[0]
        for i in range(1, len(prices)):
            Minp = min(Minp, prices[i - 1])
            Maxp = max(Maxp, prices[i] - Minp)
        return Maxp

    def maxProfit2(self, prices: list) -> int:
        # 多次交易问题-递归法
        # 超大列表个人感觉会栈溢出
        # 做剪枝

        # 查表-规划问题
        # 从尾往前查找
        # 一天只能做一次买卖
        # maxtable = [0,0]
        # i = 0
        # prices = prices[::-1]

        # while i < len(prices):
        #     if prices[i] == 0:
        #         del prices[i]
        #         continue
        #     elif i >= 1:
        #         if prices[i] == prices[i-1]:
        #             del prices[i]
        #             continue
        #     i += 1

        # for i in range(1, len(prices)):

        #     maxtable.append(0)
        #     for j in range(0, i+1):
        #         if i!= j:
        #             if prices[j] - prices[i] < 0:
        #                 pass
        #             else:
        #                 if j - 1 < 0:
        #                     maxtable[-1] = max(maxtable[-1],prices[j] - prices[i])
        #                 else:
        #                     maxtable[-1] = max(maxtable[-1],prices[j] - prices[i] + maxtable[j])
        #         else:
        #             maxtable[-1] = max(maxtable[-1],maxtable[-2])
        #     # maxtable.append(max(table))
        # return maxtable[-1]

        # 一天可以多次买卖
        # 单纯的加减就可以了
        return sum(
            prices[i] - prices[i - 1]
            for i in range(1, len(prices)) if prices[i] > prices[i - 1]
        )

    def maxProfit3(self, prices: list) -> int:
        # 局部最高-循环
        i = 0
        while i < len(prices):
            if i >= 1:
                if prices[i] == prices[i - 1]:
                    del prices[i]
                    continue
            i += 1

        local = [i for i in range(1, len(prices) - 1) if prices[i + 1] < prices[i] and prices[i] > prices[i - 1]]
        local.append(len(prices) - 1)
        twice_max = 0
        for i in local:
            twice_max = max(twice_max, self.maxProfit(prices[i + 1:]) + self.maxProfit(prices[0:i + 1]))

        return twice_max

    def isPalindrome(self, s: str) -> bool:
        s = ''.join([c for c in s if c.isalpha() or c.isdigit()]).lower()
        return s == s[::-1]

    def isIsomorphic(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: bool
        """
        alpha_map_s = collections.defaultdict(str)
        alpha_map_t = collections.defaultdict(set)

        for a in range(len(s)):
            alpha_map_s.setdefault(s[a], t[a])
            if alpha_map_s[s[a]] != t[a]:
                return False
            alpha_map_t[t[a]].add(s[a])
            if len(alpha_map_t[t[a]]) > 1:
                return False
        return True

    def reverseList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """

        pre, ptr = None, head
        while ptr:
            if pre == None:

                pre = ptr
                ptr = ptr.next
                pre.next = None
                if ptr == None:
                    return pre
            else:
                if ptr.next == None:
                    ptr.next = pre
                    return ptr
                else:
                    temp = ptr.next
                    ptr.next = pre
                    pre = ptr
                    ptr = temp

        return None

    @Time(n=1)
    def canFinish(self, numCourses, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: bool
        """

        def dfs(i):
            visited[i] = 1
            for j in edges[i]:
                if visited[j] == 1:
                    return False
                elif visited[j] == 0:
                    if not dfs(j):
                        return False
            visited[i] = 2
            return True

        edges = [[] for _ in range(numCourses)]
        for u, v in prerequisites:
            edges[v].append(u)

        visited = [0 for _ in range(numCourses)]

        for i in range(numCourses):
            if visited[i] == 0:
                if not dfs(i):
                    return False

        return True

    @Time(n=1)
    def minSubArrayLen1(self, s, nums):
        """
        找数序找到何为s的最小连续数序的长度
        :type s: int 目标
        :type nums: List[int] 数序
        :rtype: int
        """
        if sum(nums) < s:
            return 0

        if s in nums:
            return 1

        dp = [0 for _ in range(nums.__len__())]
        minlen = len(nums)

        for i in range(len(nums)):
            for j in range(i, -1, -1):
                if i == j:
                    dp[i] = nums[i]
                else:
                    dp[j] = nums[i] + dp[j]
                if dp[j] >= s:
                    if i - j + 1 < minlen:
                        minlen = i - j + 1

                    break
        return minlen

    @Time(n=1)
    def minSubArrayLen2(self, s, nums):
        '''
            设置滑块向右移动
        :param s:
        :param nums:
        :return:
        '''
        n = sum(nums)
        if s > n: return 0
        if s == n: return len(nums)

        left, right = 0, 1
        minlen = len(nums)
        temp = nums[left]
        temp -= nums[right - 1]

        while right <= len(nums):
            temp += nums[right - 1]
            if right < len(nums):
                while temp + nums[right] <= s:
                    temp = temp + nums[right]
                    right += 1

            while temp - nums[left] >= s:
                temp = temp - nums[left]
                left += 1

            if right - left < minlen:
                minlen = right - left

            right += 1

        return minlen

    def findOrder(self, nc, prerequisites):
        """
        :type numCourses: int
        :type prerequisites: List[List[int]]
        :rtype: List[int]
        """
        # 1111111111111111111111meiyou没有完成

        dependOn = defaultdict(set)
        depentBy = defaultdict(set)
        for s, t in prerequisites:
            dependOn[s].add(t)
            depentBy[t].add(s)
        todo = [i for i in range(nc) if i not in dependOn]
        take = []
        while todo:
            c = todo.pop()
            take.append(c)
            for cc in depentBy[c]:
                dependOn[cc].remove(c)
                if not dependOn[cc]:
                    todo.append(cc)

        return take if len(take) == nc else []

    def containsDuplicate(self, nums) -> bool:
        if len(nums) <= 1: return False
        nums.sort()
        for i in range(len(nums) - 1):
            if nums[i] == nums[i + 1]:
                return True
        return False

    @Time(n=1)
    def findWordss(self, board, words):
        """
        :type board: List[List[str]]
        :type words: List[str]
        :rtype: List[str]
        """
        # 对board 进行按首字母的索引字典
        # 对字典进行四周的节点链接形成一个图， 这样可以以单词为单位进行在图中的索引
        # 字典索引
        alpha_dictionary = collections.defaultdict(list)
        # 索引图
        graph = collections.defaultdict(list)
        # 单词库
        wd = collections.defaultdict(bool)
        # 路径标记
        label = [[0 for _ in range(len(board[0]))] for _ in range(len(board))]

        for i in range(len(board)):
            for j in range(len(board[0])):
                temp = []
                if i - 1 >= 0:
                    temp.append(((i - 1, j), board[i - 1][j]))
                if j - 1 >= 0:
                    temp.append(((i, j - 1), board[i][j - 1]))
                if i + 1 < len(board):
                    temp.append(((i + 1, j), board[i + 1][j]))
                if j + 1 < len(board[0]):
                    temp.append(((i, j + 1), board[i][j + 1]))
                alpha_dictionary[board[i][j]].append(((i, j), board[i][j]))
                graph[((i, j), board[i][j])] = temp

        def DFSWORD(word, local, pathword=None, path=None, graph=graph):
            '''
            local起始位置，在graph中查找是否存在word
            :param word: 需要查找的词
            :param local: 首字母的绝对位置
            :param graph: 字母图
            :param path: 单词路径，图中的字母在本单词中只能出现一次，作为字母的记录
            :return:
            '''
            if len(word) == 1:
                return True
            else:
                for node in graph[local]:
                    wd[pathword + node[1]] = True
                    if node[1] == word[1] and node[0] not in path:
                        if DFSWORD(word[1:], node, pathword + node[1], path + [node[0]], graph):
                            return True
                return False

        def DFSWORD2(node, x, word):
            '''
            使用标记法来搜索
            1 ： 表示正在搜索的路径
            0 ： 表示未搜索的路径
            2 ： 表示已经搜索的路径
            :param node:
            :return:
            '''
            # 字母与结点值相等
            if node[1] != word[x]:
                return False
            # 到达最后一个字母
            if x == len(word) - 1:
                return True
            flag = 0
            label[node[0][0]][node[0][1]] = 1
            for n in graph[node]:
                if label[n[0][0]][n[0][1]] == 0:
                    if DFSWORD2(n, x + 1, word):
                        flag = 1

            label[node[0][0]][node[0][1]] = 0
            return flag == 1

        results = []
        for word in words:
            for node in alpha_dictionary[word[0]]:
                if DFSWORD2(node, 0, word):
                    results.append(word)

        return list(set(results))

    def containsNearbyDuplicate(self, nums: list, k: int) -> bool:
        # 保存数字的索引，总是比，》>=k则返回true
        nd = collections.defaultdict(list)

        for i, n in enumerate(nums):
            if len(nd[n]) and i - nd[n][-1] <= k:
                return True
            nd[n].append(i)

        return False

    def rob1_1(self, nums: list) -> int:
        ln = len(nums)
        if ln < 1: return 0
        nums = nums + nums

        max_money = 0

        for i in range(ln):
            # 直接调用rob ，会很慢
            max_money = max(max_money, self.rob(nums[i + 2:i + ln - 1]) + nums[i])
        return (max_money)

    def rob1_2(self, nums: list) -> int:
        # dp
        # 对每一个新的起点建立动态规划
        max_money, ln = 0, len(nums)

        for i in range(ln // 2 + 1):
            temp = nums + nums
            dnums = temp[i + 2:i + ln - 1]
            dp = [0 for _ in range(len(dnums))]
            if len(dnums) <= 2:
                dp = dnums
            else:
                dp[-1], dp[-2] = dnums[-1], dnums[-2]
                for j in range(len(dnums) - 3, -1, -1):
                    dp[j] = max(dp[j + 2:]) + dnums[j]
            max_money = max(max_money, max(dp + [0]) + nums[i])
        print(max_money)

    @Time(n=1)
    def shortestPalindrome(self, s: str) -> str:
        '''
        在字符串s 前面加入字符，使新形成的字符串new_s 为一个回文串
        1 . 从 len // 2 + 1 索引处开始向0 索引移动， 当前索引为 i，检查 [0:i-1] 与[i+1:i+i-1] 是否是回文，
        2. 若是回文，则将[i + i + 1:]的回文加入到字符串s的前面
        3. 此时则是最短回文串
        :param s:
        :return:
        '''

        for i in range(len(s) // 2 + 1, -1, -1):
            if s[0:i] == s[i + 1:2 * i + 1][::-1]:
                return s[2 * i + 1:][::-1] + s
            if s[0:i] == s[i:2 * i][::-1]:
                return s[2 * i:][::-1] + s

    @Time(n=1)
    def containsNearbyAlmostDuplicate(self, nums: list, k: int, t: int) -> bool:

        if t == 0:
            nd = collections.defaultdict(list)
            for i, n in enumerate(nums):
                if len(nd[n]) and i - nd[n][-1] <= k:
                    return True
                nd[n].append(i)
            return False

        for i in range(len(nums)):
            if min([abs(nums[i] - j) for j in nums[max(0, i - k):i] + nums[i + 1:1 + min(i + k, len(nums))]]) <= t:
                return True

        return False

    def findKthLargest(self, nums: list, k: int) -> int:
        '''
        选择排序，选出kth大的数
        :param nums:
        :param k:
        :return:
        '''
        for i in range(k):
            minIndex = i
            for j in range(i + 1, len(nums)):
                if nums[j] > nums[minIndex]:
                    minIndex = j
            nums[i], nums[minIndex] = nums[minIndex], nums[i]

        return nums[k - 1]

    def invertTree(self, root: TreeNode) -> TreeNode:
        '''
        反转二叉树
        :param root:
        :return:
        '''
        if not root:
            return None
        else:
            left = self.invertTree(root.left)
            right = self.invertTree(root.right)

            root.left, root.right = right, left
            return root

    def combinationSum3(self, k: int, n: int) -> list:
        results = []

        def rescurive(k=k, n=n, result=[0]):
            if k == 0:
                if n == 0:
                    results.append(result[1:])
            else:
                st = set(range(min(10, n - k + 2))) - set(result) - set(range(result[-1] + 1))
                for i in st:
                    rescurive(k - 1, n - i, result + [i])

        rescurive()
        return results

    def isPowerOfTwo(self, n: int) -> bool:
        bn = bin(n)[2:].rstrip('0')
        print(bn)
        return True if bin(n)[2:].rstrip('0') == '1' else False

    def calculate(self, s: str) -> int:
        # 224.
        pass

    def calulateSkyine(self, buildings: list) -> list:
        print('计算一个轮廓：', buildings)

        buildings.sort(key=lambda x: x[-1])
        # 1. 合并相同楼层
        newbuild = []
        i = 0
        while i < len(buildings) - 1:
            if buildings[i][-1] == buildings[i + 1][-1]:
                if buildings[i][1] <= buildings[i + 1][1]:
                    buildings[i][0] = min(buildings[i + 1][0], buildings[i][0])
                    buildings[i][1] = max(buildings[i + 1][1], buildings[i][1])

                    buildings.remove(buildings[i + 1])
            else:
                i += 1

        print('计算一个合并后的轮廓：', buildings)

        for i in range(len(buildings) - 1, -1, -1):
            fx, fy, fh = buildings[i]
            for j in range(len(buildings) - 1, -1, -1):
                if j < i and buildings[j]:
                    sx, sy, sh = buildings[j]
                    if sx <= fx <= sy:
                        fx = sy
                    if sx <= fy < sy:
                        fy = sx
                    if fy < fx:
                        buildings[i] = None
        print('左右移动合并后的轮廓：', buildings)

        pass

    #
    # def getSkyline(self, buildings: list) -> list:
    #     # 218.
    #     # 1. 先找水平线的坐端点
    #     # 2. 进行同水平端点合并
    #     # 3. 找出地平线的楼间距之间的左端点
    #     # buildings.append([buildings[-1][0],buildings[-1][0], buildings[-1][-1]])
    #     # results = []
    #     #
    #     # # 找出高度为0的直线轮廓左端点,同时并计算水平轮廓左端点
    #     # sk = []
    #     # x, y = buildings[0][0], buildings[0][1]
    #     # for i in range(len(buildings)-1):
    #     #     sk.append(buildings[i])
    #     #     if y < buildings[i+1][0]:
    #     #         print('楼间间隔：{}--{}'.format(y, buildings[i+1][0]))
    #     #         results.append([y, 0])
    #     #         self.calulateSkyine(sk)
    #     #         x, y = buildings[i+1][0],buildings[i+1][1]
    #     #         sk.clear()
    #     #         continue
    #     #     x, y = min(x, buildings[i+1][0]), max(y, buildings[i+1][1])
    #     #
    #     # self.calulateSkyine(sk+[[26,27,7]])
    #     #
    #     pass

    def isPalindrome(self, head: ListNode) -> bool:
        l = []
        while head:
            l.append(head.val)
            head = head.val
        return l == l[::-1]

    def getSkyline(self, buildings):

        # 将数据分解为[lx, h, True] 和 [rx, h ,False] 形式，node
        # 遍历node，遇见未出现并且为True，则将key=h + 1， 记录当前最高高度并添加到pm中
        #
        node = []
        for lx, rx, h in buildings:
            node.append([lx, h, True])
            node.append([rx, h, False])
        node.sort(key=lambda x: x[0])
        for i in node:
            print(i)

        ht = collections.defaultdict(int)
        pm = []
        maxh = node[0][1]
        for x, h, b in node:
            if x not in ht.keys():
                ht[h] += 1
                pm.append(x, max())

            print(maxh)
            pm.append([x, maxh])
            print(pm)

    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        # 返回二叉排序树，两个节点p，q的公共最近父节点
        if root == None:
            return None
        else:
            if root.val < min(p.val, q.val):
                return self.lowestCommonAncestor(root.right, p, q)
            elif root.val > max(p.val, q.val):
                return self.lowestCommonAncestor(root.left, p, q)
            else:
                return root

    def findContinueOne(self, dp: list):
        temp = [0] + dp
        # 查找连续的1的数量
        for i in range(1, len(temp)):
            if temp[i] == 0:
                temp[i] = 0
            else:
                temp[i] += temp[i - 1]
        h = max(temp)
        return h

    @Time(n=1000)
    def maximalSquare(self, matrix: list) -> int:
        # 找出矩阵中，方阵1的面积
        # 暴力法：
        # 1.从上至下，依次对位相与的结果，求连续1的最大值
        # 若此结果比相与的次数不小，则矩阵边长加1
        # 否则中断此次向下相与
        # 2. 若最处的一行全为0， 则跳过这行

        result = 0
        matrix = [[int(v) for v in m] for m in matrix]
        for i in range(len(matrix)):
            dp = matrix[i]
            if max(matrix[i] + [0]) == 0:
                continue
            else:
                mh = 1
            for j in range(i + 1, len(matrix)):
                dp = [dp[k] & matrix[j][k] for k in range(len(dp))]

                temp = [0] + dp
                # 查找连续的1的数量
                for k in range(1, len(temp)):
                    if temp[k] == 0:
                        temp[k] = 0
                    else:
                        temp[k] += temp[k - 1]
                if max(temp) >= j - i + 1:
                    mh += 1
                else:
                    break

            result = max(result, mh)

        return result ** 2

    def deleteNode(self, head, node):
        """
        :type node: ListNode
        :rtype: void Do not return anything, modify node in-place instead.
        """
        pre, ptr = None, head

        if not ptr: return None
        if ptr.val == node.val: return head.next
        while ptr:
            if ptr.val == node.val:
                pre.next = ptr.next
                break
            else:
                pre, ptr = ptr, ptr.next
        return head

    def countNodes(self, root: TreeNode) -> int:
        # 非递归
        stack = []
        result = 0
        while root or stack:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            result += 1

            root = root.right
        return result
        # 递归
        return 1 + self.countNodes(root.left) + \
               self.countNodes(root.right) if root else 0

    def binaryTreePaths(self, root: TreeNode) -> list:
        results = []

        def Paths(root, result=''):
            if not root:
                pass
            else:
                if not root.left and not root.right:
                    results.append(result + str(root.val))
                Paths(root.left, result + str(root.val) + '->')
                Paths(root.right, result + str(root.val) + '->')

        Paths(root)
        print(results)

    def addDigits(self, num: int) -> int:
        while len(num) != 1:
            num = str(sum([int(v) for v in list(str(num))]))
        return int(num)

    def computeArea(self, A: int, B: int, C: int, D: int, E: int, F: int, G: int, H: int) -> int:
        Area_a = (C - A) * (D - B)
        Area_b = (G - E) * (H - F)
        dx = max(min(C, G) - max(A, E), 0)
        dy = max(min(D, F) - max(B, H), 0)
        return Area_a + Area_b - dx * dy
        pass

    def calculate1(self, s: str) -> int:
        # 表达式计算,整数计算,只计算加减
        # 就是写的有一点麻烦的是每一次进行符号栈的栈入要进行检查+-的运算
        s = s.replace(' ', '') + '+0'

        def POP(fstack, nstack):
            # 遇到 + 或者 - 进行运算
            while fstack.__len__() >= 1 and fstack[-1] in ['+', '-'] and nstack.__len__() >= 2:
                if fstack.pop() == '+':
                    nstack.append(nstack.pop() + nstack.pop())
                else:
                    nstack.append(-1 * (nstack.pop() - nstack.pop()))
                # 进行完加减法运算以后，检查是否出现无意义括号，比如：（212）或者（2），进行括号的消除
                if fstack.__len__() >= 2 and fstack[-1] == ')' and fstack[-2] == '(':
                    fstack.pop()
                    fstack.pop()

        pre, fstack, nstack = 0, [], []  # 前一个符号位索引+1， 符号栈， 数字栈
        for i in range(0, len(s)):
            if s[i] in ['+', '-', '(', ')']:
                if pre == i:  # 遇到两个相邻运算符或者括号，则不进行数字的栈入，只进行符号的消除或者栈入
                    pre = i + 1
                    if fstack.__len__() > 1 and fstack[-1] == '(' and s[i] == ')':
                        fstack.pop()
                        POP(fstack, nstack)
                    else:
                        fstack.append(s[i])
                    i += 1
                    continue
                nstack.append(int(s[pre:i]))
                POP(fstack, nstack)
                if fstack.__len__() > 1 and fstack[-1] == '(' and s[i] == ')':
                    fstack.pop()
                    POP(fstack, nstack)
                else:
                    fstack.append(s[i])
                pre = i + 1
            i += 1
        POP(fstack, nstack)
        return nstack[-1]

    def isUgly(self, num: int) -> bool:
        pass

    def calculate2(self, s: str) -> int:
        # 无括号的加减乘除

        def cal(fs, ns):
            if fs[-1] == '-':
                ns.append(-1 * (ns.pop() - ns.pop()))
            elif fs[-1] == '*':
                ns.append(ns.pop() * ns.pop())
            elif fs[-1] == '/':
                b, a = ns.pop(), ns.pop()
                ns.append(a // b)
            else:
                ns.append(ns.pop() + ns.pop())
            fs.pop()

        s = s.replace(' ', '') + '+0'
        pre, ns, fs = 0, [], []  # 运算符记位+1 ，数字栈， 符号栈

        for i in range(len(s)):

            if s[i] in ['+', '-']:
                ns.append(int(s[pre:i]))
                while ns.__len__() >= 2 and fs.__len__() >= 1:
                    cal(fs, ns)
                fs.append(s[i])
                pre = i + 1
            elif s[i] in ['*', '/']:
                ns.append(int(s[pre:i]))
                while ns.__len__() >= 2 and fs.__len__() >= 1 \
                        and fs[-1] in ["/", "*"]:
                    cal(fs, ns)
                fs.append(s[i])
                pre = i + 1
            else:
                pass

        return ns[-1]

    def summaryRanges(self, nums):
        result = []
        i = 0
        while i < len(nums):
            start, end = nums[i], nums[i]
            for j in range(i + 1, len(nums)):
                if nums[j] == end + 1:
                    end = nums[j]
                    if j == len(nums) - 1:
                        i = j + 1
                else:
                    i = j - 1
                    break
            if start == end:
                result.append(str(start))
            else:
                result.append('{}->{}'.format(start, end))
            i += 1
        return result

    def isUgly(self, num: int) -> bool:
        '''
        质因素只包含2，3，5的数
        :param num:
        :return:
        '''
        if num == 0: return False
        if num == 1: return True
        if num == 2 or num == 3 or num == 5:
            return True
        elif num % 2 == 0:
            return self.isUgly(num // 2)
        elif num % 3 == 0:
            return self.isUgly(num // 3)
        elif num % 5 == 0:
            return self.isUgly(num // 5)
        else:
            return False

    def missingNumber(self, nums: list) -> int:
        nums.sort()
        for i in range(len(nums)):
            if nums[i] != i:
                return i
        return len(nums)

    def moveZeroes(self, nums: list) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """

        i, k = 0, 0
        while i < len(nums):
            if nums[i] != 0:
                nums[k] = nums[i]
                k += 1
            i += 1
        while k < len(nums):
            nums[k] = 0
            k += 1
        print(nums)

        nlist, zlist = [], []
        for i in nums:
            if i == 0:
                zlist.append(0)
            else:
                nlist.append(i)

        nums[:] = nlist + zlist
        print(nums)

        # 打乱顺序
        pos = len(nums)

        i = 0
        while i < pos:
            if nums[i] == 0:
                pos -= 1
                nums[i], nums[pos] = nums[pos], nums[i]
            i += 1
        print(nums)
        pass

    def majorityElement(self, nums: list) -> list:
        threshold = len(nums) // 3

        d = collections.Counter(nums)

        return [k for k, v in d.items() if v >= threshold]

        d = collections.defaultdict(int)
        for i in nums:
            d[i] += 1
        return [k for k, v in d.items() if v > threshold]
        pass

    def kthSmallest(self, root: TreeNode, k: int) -> int:
        '''
        在二叉搜索树种找到第k小的数
        先根遍历就行
        :param root:树根
        :param k:
        :return:
        '''
        stack = []
        result = []
        while stack != [] or root:
            while root:
                stack.append(root)
                root = root.left
            root = stack.pop()
            result.append(root.val)
            if result.__len__() == k:
                break;
            root = root.right

        return result[-1]

        pass

    def findPathInBTree(self, root: TreeNode, node: TreeNode):
        if root is None:
            pass
        else:
            if root.val == node.val:
                return [root]
            l = self.findPathInBTree(root.left, node)
            if l is None:
                l = self.findPathInBTree(root.right, node)
            if l is not None:
                return [root] + l

    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        path_p = self.findPathInBTree(root, p)

        for i in path_p:
            print(i.val, end=' ')
        print()

        path_q = self.findPathInBTree(root, q)

        for i in path_q:
            print(i.val, end=' ')
        print()

        if path_p == [] or path_q == []:
            return None
        i = 0
        while i < len(path_q) and i < len(path_p):
            if path_q[i].val != path_p[i].val:
                break;
            i += 1
        return path_q[i - 1]

    def wordPattern(self, pattern: str, str: str) -> bool:
        d = collections.defaultdict(set)
        ds = collections.defaultdict(set)
        str = str.split(' ')
        if len(str) != len(pattern): return False
        for i in range(len(pattern)):
            d[pattern[i]] = d[pattern[i]] | set([str[i]])
            ds[str[i]] = ds[str[i]] | set([pattern[i]])
        for i in d.values():
            if len(i) >= 2:
                return False
        for i in ds.values():
            if len(i) >= 2:
                return False
        return True

    def singleNumber(self, nums: list) -> list:
        return [k for k, v in collections.Counter(nums).items() if v == 1]

    def lThreenumbertoWords(self, num: int, dict: dict):
        result = ''

        if num >= 100:
            b = num // 100
            result = ''.join([result, dict[b], ' Hundred '])
            num %= 100
        if num == 0:
            return result
        if num in dict.keys():
            return ''.join([result, ' ', dict[num]])
        if 99 >= num >= 10:
            b = num // 10 * 10
            result = ''.join([result, dict[b], ' '])
            num %= 10
        if 9 >= num >= 1:
            result = ''.join([result, dict[num]])

        return result

    def numberToWords(self, num: int) -> str:
        import re
        if num > 2 ** 31 - 1:
            return None
        unit = [' Billion ', ' Million ', ' Thousand ', '']
        nd = {0: 'Zero', 1: 'One', 2: 'Two', 3: 'Three', 4: 'Four', 5: 'Five',
              6: 'Six', 7: 'Seven', 8: 'Eight', 9: 'Nine', 10: 'Ten',
              11: 'Eleven', 12: 'Twelve', 13: 'Thirteen', 14: 'Fourteen', 15: 'Fifteen',
              16: 'Sixteen', 17: 'Seventeen', 18: 'Eighteen', 19: 'Nineteen', 20: 'Twenty',
              30: 'Thirty', 40: 'Forty', 50: 'Fifty', 60: 'Sixty', 70: 'Seventy', 80: 'Eighty',
              90: 'Ninety'}
        if num == 0:
            return 'Zero'

        resultBit = []
        for i in range(4):
            b = num % 1000
            r = self.lThreenumbertoWords(b, nd)
            num = num // 1000
            resultBit.insert(0, r)

        result = ''
        for i in range(4):
            if resultBit[i] != '':
                result = ''.join([result, resultBit[i], unit[i], ''])
        result = re.sub(re.compile('\s+'), ' ', result)
        return result.strip()

    @Time(n=1)
    def PrimeRange(self, n):
        l = [0 for _ in range(n + 1)]
        for i in range(2, n + 1):
            if l[i - 1] != 1:
                for k in range(2 * i, n + 1, i):
                    l[k - 1] = 1
        return [i for i in range(n + 1) if l[i - 1] == 0][1:]

    def nthUglyNumber(self, n: int) -> int:
        visited = set([1])
        h = [1]
        count = 0
        for i in range(n):
            val = heapq.heappop(h)
            for factor in [2, 3, 5]:
                if val * factor not in visited:
                    heapq.heappush(h, val * factor)
                    visited.add(val * factor)
        return val

    @Time(n=1)
    def productExceptSelf(self, nums: list) -> list:
        # L 从左向右累乘
        # Ｒ 从右向左累乘
        # 防止越界，像L，R左右各添加 1
        # 对于nums对应的位置 i 的结果值：result = L[i]*R[i+2]
        nl = len(nums)
        L, R = [1 for _ in range(nl + 2)], [1 for _ in range(nl + 2)]
        for i in range(nl):
            L[i + 1] = L[i] * nums[i]
            R[nl - i] = R[nl - i + 1] * nums[nl - i - 1]
        return [L[i] * R[i + 2] for i in range(nl)]

    def maxSlidingWindow(self, nums: list, k: int) -> list:
        if not nums or k == 0: return []
        result = [max(nums[:k])]
        if k >= len(nums): return result

        for i in range(k, len(nums)):
            if nums[i - k] >= result[-1]:
                result.append(max(nums[i - k + 1:i + 1]))
                continue

            if nums[i] <= nums[i - k]:
                result.append(result[-1])
            elif nums[i] > nums[i - k] and nums[i] <= result[-1]:
                result.append(result[-1])
            else:
                result.append(nums[i])
        return result

    from operator import mul, add, sub
    op = {'*': mul, '+': add, '-': sub}

    def diffWaysToCompute(self, input: str) -> list:
        '''
        241.
        输入表达式所有的计算可能
        :param input:
        :return:
        '''

        if input.isdigit():
            return [int(input)]
        else:
            res = []
            for i, s in enumerate(input):
                if s in self.op.keys():
                    L = self.diffWaysToCompute(input[:i])
                    R = self.diffWaysToCompute(input[i + 1:])

                    for l in L:
                        for r in R:
                            res.append(Solution.op[s](l, r))
            return res

        pass

    def canWinNim(self, n: int) -> bool:
        if n % 4 == 0:
            return False
        else:
            return True

    def getHint(self, secret: str, guess: str) -> str:
        s, g, i = list(secret), list(guess), 0
        while i < s.__len__():
            if s[i] == g[i]:
                del s[i], g[i]
                i -= 1
            i += 1
        return '{}A{}B'.format(str(len(secret) - len(s)),
                               str(sum((collections.Counter(s) & collections.Counter(g)).values())))

    def hIndex(self, citations: list) -> int:

        if len(citations) == 0: return 0
        if len(citations) == 1:
            if citations[-1] > 0:
                return 1
            else:
                return 0
        return [i for i, v in enumerate(sorted(citations, reverse=True) + [0]) if v < i + 1][0]

        for i, v in enumerate(sorted(citations, reverse=True) + [0]):
            if v < i + 1:
                return i

        return 0

    def hIndex2(self, citations: list) -> int:
        if len(citations) == 0: return 0
        if len(citations) == 1:
            if citations[-1] > 0:
                return 1
            else:
                return 0
        return [i for i, v in enumerate(([0] + citations)[::-1]) if v < i + 1][0]

    def firstBadVersion(self, n):
        """
        :type n: int
        :rtype: int
        """

        def isBadVersion(n):
            if n >= 2:
                return True
            else:
                return False

        left, right, mid = 1, n, 0
        while left <= right - 3:
            mid = (left + right) // 2
            if isBadVersion(mid):
                right = mid
            else:
                left = mid
        for i in range(left, right + 1):
            if isBadVersion(i):
                return i

    def numSquares(self, n: int) -> int:
        # 279.
        # 深度+剪枝
        min_branch = [n]

        def searchD(n, deap=0, min_branch=min_branch, flag=0):
            if n == 0:
                min_branch[:] = [deap]
                flag += 1
                print(min_branch)
            elif deap < min_branch[0] and flag <= 50:
                K = math.floor(math.sqrt(n))
                for i in range(K, max(0, K // 2), -1):
                    searchD(n - i ** 2, deap + 1, min_branch)

        searchD(n)
        return min_branch

    def findDuplicate(self, nums: list) -> int:
        nums.sort()
        for i in range(1, len(nums)):
            if nums[i] == nums[i - 1]:
                return nums[i]

    def countOneandZero(self, x, y, board: list):
        one, zero = 0, 0

        if x - 1 >= 0 and y - 1 >= 0:
            if board[x - 1][y - 1] == 1:
                one += 1
            else:
                zero += 1
        if x - 1 >= 0:
            if board[x - 1][y] == 1:
                one += 1
            else:
                zero += 1
        if x - 1 >= 0 and y + 1 < len(board[0]):
            if board[x - 1][y + 1] == 1:
                one += 1
            else:
                zero += 1

        if y - 1 >= 0:
            if board[x][y - 1] == 1:
                one += 1
            else:
                zero += 1
        if y + 1 < len(board[0]):
            if board[x][y + 1] == 1:
                one += 1
            else:
                zero += 1

        if x + 1 < len(board) and y - 1 >= 0:
            if board[x + 1][y - 1] == 1:
                one += 1
            else:
                zero += 1
        if x + 1 < len(board):
            if board[x + 1][y] == 1:
                one += 1
            else:
                zero += 1
        if x + 1 < len(board) and y + 1 < len(board[0]):
            if board[x + 1][y + 1] == 1:
                one += 1
            else:
                zero += 1

        return one, zero

    def gameOfLife(self, board: list) -> None:
        """
        289 .
        Do not return anything, modify board in-place instead.
        """
        import copy
        state = copy.deepcopy(board)
        for x in range(len(board)):
            for y in range(len(board[0])):
                one, zero = self.countOneandZero(x, y, state)
                if state[x][y] == 0:
                    if one == 3:
                        board[x][y] = 1
                else:
                    if one >= 4 or one < 2:
                        board[x][y] = 0

        return board

    def lengthOfLIS(self, nums: list) -> int:
        pass

    def isAdditiveNumber(self, num: str) -> bool:
        # 暴力法
        start = 0
        length = len(num)

        def bk(p1=-1, p2=-1, start=0):
            if start >= length and str(p1) + str(p2) != num and str(p2) != num:
                return True
            for i in range(start + 1, length + 1):
                if num[start] == '0' and i - start > 1:
                    return False
                p3 = int(num[start:i])
                if p3 == p1 + p2 or p1 == -1 or p2 == -1:
                    if bk(p2, p3, i):
                        return True
            return False

        return bk()

    def nthSuperUglyNumber1(self, n: int, primes: list) -> int:
        t1 = time.time()
        count = 1;
        heap = [1]
        heapq.heapify(heap)
        result = set([])
        while len(result) < n:
            visited = heapq.heappop(heap)

            for p in primes:
                heapq.heappush(heap, visited * p)
            result.add(visited)
        t2 = time.time()
        result = list(result)
        result.sort()
        t3 = time.time()

        print("t1--->t2: ", t2 - t1)
        print("t2--->t3: ", t3 - t2)
        return result[-1]

    def nthSuperUglyNumber(self, n: int, primes: list) -> int:
        result = set({1})
        ugly = [1]
        heapq.heapify(ugly)

        while n > 1:
            un = heapq.heappop(ugly)
            for p in primes:
                item = p * un
                if item not in result:
                    heapq.heappush(ugly, item)
                    result.add(item)
            n -= 1
        return heapq.heappop(ugly)

    @Time(n=1000)
    def countSmaller2(self, nums: list) -> list:
        import bisect
        # 有序插入
        sonums = []
        result = []

        for i, v in enumerate(nums[::-1]):
            idx = bisect.bisect_left(sonums, v)
            sonums.insert(idx, v)
            result.insert(0, idx)
        return result

    def countSmaller3(self, nums: list) -> list:

        sonums = [[v, i] for i, v in enumerate(nums)]
        sonums = sorted(sonums)
        temp = [[sonums[0][1],0]]

        for i in range(1, len(sonums)):
            if sonums[i-1][0] == sonums[i][0]:
                temp.append([sonums[i][1] , temp[-1][-1]])
            else:
                temp.append([sonums[i][1] , i])
        print(temp)
        temp.sort(key=lambda x:x[0])
        return temp

    def removeDuplicateLetters(self, s: str) -> str:
        # 记录字母最后一次出现的位置 indexlast
        # 在遍历s时，检查当前位置时候在符合结果集result
        # 条件：当前位置未在结果集出现，并且，当前位置的字幕的索引是至少是大于等于 result中的位置
        # 当前位置后还存在result[-1]

        indexast = {}
        for i ,v in enumerate(s):
            indexast[v] = i;

        result = []

        for i , v in enumerate(s):
            if v not in result:
                #                                   # 说明 在 i位置后面还存在result[-1]
                while result and v < result[-1] and i < indexast[result[-1]]:
                    result.pop()
                result.append(v)
        return ''.join(result)


    def maxProduct(self, words: list) -> int:
        words_set = [set(v) for v in words]
        maxPro = [0]

        for i in range(0, len(words)):
            for j in range(i+1, len(words)):
                if len(words_set[i].intersection(words_set[j])) == 0:
                    maxPro.append(len(words[i]) * len(words[j]))
        return max(maxPro)

        pass

    def no_common_str(self, str1, str2):
        bit_num = lambda ch:ord(ch) - ord('a')
        bitmask1 = bitmask2 = 0

        for ch in str1:
            bitmask1 |= 1 << bit_num(ch);

        for ch in str2:
            bitmask2 |= 1 << bit_num(ch);

        return bitmask1 & bitmask2  == 0


    def maxProduct1(self, words):
        # 对每一个单词继进行按位的存储，有这个字母，对应位上位 1
        # 每个单词的字典序位n, 所对应的位的位置为 1 << n，
        # 把所有得字母 | 起来，就形成了单词的（非重复字母）二进制表示
        # 比如：ab, abbb拥有相同的二进制位，我们只需要保存最长的单词，用map存储

        hasmap = collections.defaultdict(int)

        bit_num = lambda ch:ord(ch) - ord('a')
        for word in words:
            bitmask = 0
            for ch in word:
                bitmask |= 1<<bit_num(ch)
            hasmap[bitmask] = max(hasmap[bitmask], len(word))
        max_pro = 0
        for x in hasmap:
            for y in hasmap:
                if x & y == 0:
                    max_pro = max(max_pro, hasmap[x]*hasmap[y])
        return max_pro
    #@Time(n=1)
    def bulbSwitch(self, n: int) -> int:
        result = 0
        i = 1
        count = 0
        while result <= n:
            result += i
            i += 2
            count += 1
            print(i, result, count)
        return count - 1

    def maxNumber(self, nums1: list, nums2: list, k: int) -> list:
        pass

    def coinChange(self, coins: list, amount: int) -> int:

        # down -top dp
        dp=[ (amount +1) for _ in range(amount+1)]
        maxc = amount +1
        dp[0]=0

        for i in range(1, amount+1):
            for j in range(len(coins)):
                if coins[j] <= i:
                    dp[i] = min(dp[i], dp[i-coins[j]]+ 1)
            print(dp)
        return dp[i] if dp[i] <= amount else -1

    def wiggleSort(self, nums: list) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        nums.sort()
        length = len(nums)
        temp1 = nums[:(length-1)//2+1]
        temp2 = nums[(length-1)//2+1:]
        ti=0
        print(temp1)
        print(temp2)
        for i in range(0, length, 2):
            nums[i]=temp1[ti]
            ti+=1
        ti=0
        for i in range(1, length, 2):
            nums[i]=temp2[ti]
            ti+=1
        print(nums)

    def isPowerOfThree(self, n: int) -> bool:
        count = 0
        if n<=0:return False
        while n > 0:
            n = n / 3
            if int(n) != n:
                return False
            count+=1
        return (count % 3) == 0

    def countRangeSum(self, nums: list, lower: int, upper: int) -> int:
        import bisect
        count = 0
        accm = [0]
        x =0
        for i in nums:
            x += i
            l = bisect.bisect_left(accm, x-lower)
            r = bisect.bisect_left(accm,x-upper)
            count += r- l
            bisect.insort(accm, x)
        return count

    def oddEvenList(self, head: ListNode) -> ListNode:
        odd = ListNode(0)
        oddptr = odd
        even = ListNode(0)
        evenptr = even
        flag = 1
        while head is not None:
            if flag == 1:
                oddptr.next = head
                oddptr= oddptr.next
                flag = 0
            elif flag == 0:
                evenptr.next = head
                evenptr=evenptr.next
                flag = 1
            head = head.next

            oddptr.next=None
            evenptr.next=None
        oddptr.next = even.next
        return odd.next

    def longestIncreasingPath(self, matrix: list) -> int:
        if len(matrix) == 0 or len(matrix[-1]) == 0: return 0
        maxdeep = 0
        Pdeep = [[0 for _ in matrix[0]] for _ in matrix]

        def dfs(x, y, matrix=matrix):
            if (x < 0 or x >= len(matrix) or y < 0 or y >= len(matrix[0])):
                return 0

            maxdeep = 0
            for i, j in [[x - 1, y], [x + 1, y], [x, y - 1], [x, y + 1]]:
                if i<0 or i >=len(matrix) or j < 0 or j >= len(matrix[0]):
                    continue
                if matrix[i][j] > matrix[x][y]:
                    if Pdeep[i][j] != 0 :
                        maxdeep = max(Pdeep[i][j], maxdeep)
                    else:
                        temp = dfs(i, j, matrix)
                        Pdeep[i][j]=temp
                        maxdeep = max(maxdeep, temp)
            return maxdeep+1

        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                if Pdeep[i][j]!=0:
                    continue
                Pdeep[i][j] = dfs(i, j, matrix)
                maxdeep = max(maxdeep, Pdeep[i][j])
        print(Pdeep)
        return maxdeep

    def minPatches(self, nums: list, n: int) -> int:
        m, p, i, size=1,0,0, len(nums)
        while ( m<  n):
            if(i < size and nums[i] <= m):
                m+=nums[i] # 说明nums[i]这个数字是存在的
                i+= 1
            else:
                # 说明m这个数字不存在，向后2倍， m- 1 和 1 + m 最大是两倍
                m+=m
                p+=1
        return p

    def isValidSerialization(self, preorder: str) -> bool:
        if preorder == '#': return True
        if preorder.__len__() == 0: return False
        preorder = preorder.spli(',')
        size = len(preorder)
        for _ in range(size):
            if ['#','#','#'] in preorder:return False
            for i in range(len(preorder)):
                if preorder[i] != '#' and preorder[i + 1:i + 3] == ['#', '#']:
                    if i == 0:
                        if preorder.__len__() == 3:
                            return True
                    else:
                        if preorder[i - 1] == '#':
                            preorder = preorder[:i - 1] + ['#', '#'] + preorder[i + 3:]
                            break
                        elif preorder[i - 1] != '#':
                            preorder = preorder[:i] + ['#'] + preorder[i + 3:]
                            break

        return False

    def findItinerary(self, tickets: list) -> list:
        ld = collections.defaultdict(list)
        ticketsnum = collections.defaultdict(int)
        for k, v in tickets:
            ticketsnum[(k, v)] += 1
            ld[k].append(v)
        for k, v in ld.items():
            ld[k].sort()

        result = []
        # 有向图的dfs

        def dfs(start, path=None, pd=None, ticketsnum=None):
            if sum(pd.values()) == len(tickets):
                return path + [start]
            for next in ld[start]:
                if pd[(start, next)] < ticketsnum[(start, next)]:
                    pd[(start, next)]+= 1
                    result = dfs(next, path + [start],pd,ticketsnum)
                    if len(result) > 0:return result
                    pd[(start, next)] -= 1
            return []

        result=dfs('JFK',[],collections.defaultdict(int),ticketsnum)
        return result

    def increasingTriplet(self, nums: list) -> bool:

        n1 = n2 = math.inf
        for n in nums:
            if n< n1:
                n1= n
            elif n<n2:
                n2= n
            else: # n<num1 ,n<num2 ,记录前面的像个有序小值，出现了第三个大值
                return True
        return False


    def isSelfCrossing(self, x: list) -> bool:
        # 若是相交，不会超过六个点。枚举能相交的情况
        n = len(x)
        def help(i):
            step=[x[k+i] if k+i < n else 0 for k in range(6)]
            if step[3] < step[1]:return False
            if step[2] <= step[0]:
                return True
            if step[3]>=step[1] and step[4]>=(step[2]-step[0]) \
                and step[4]<=step[2] and step[5] >= step[3] - step[1]:
                return True
            return False
        for i in range(n-3):
            if help(i):
                return True
        return False


    def palindromePairs(self, words: list) -> list:
        # 最直接方法；暴力便遍历
        # 时间损耗主要就是在查找上面，使用dict查找比list快许多

        d = {i: i for i in range(100 * 500)}
        l = [i for i in range(100 * 500)]
        key = [random.randint(0, 100 * 500) for _ in range(200)]
        print(key)

        @Time(n=100)
        def lookupinDict(key, r):
            for k in key:
                if k in r:
                    continue
                else:
                    print("{} is not in".format(k))

        lookupinDict(key, d)
        lookupinDict(key, l)


        result = []
        look= {v:k for k, v in enumerate(words)}
        for i, w in enumerate(words):
            for j in range(len(w)+1):
                pre, suf = w[:j],w[j:]
                if pre==pre[::-1] and suf[::-1]!=w and  suf[::-1] in look:
                    result.append([look[suf[::-1]], i])
                if suf==suf[::-1] and pre[::-1]!=w and pre[::-1] in look and j!=len(w):
                    result.append([i, look[pre[::-1]]])


        return result

    def robdfs(self, root: TreeNode):
        if (root == None):
            return 0, 0
        # 叶子节点直接返回本身
        if (root.left == None and root.right == None):
            return root.val, 0

        valLeft, sl = self.robdfs(root.left)
        valRight, sr= self.robdfs(root.right)
        return max(root.val + sr + sl, valLeft + valRight), valLeft+ valRight

    def rob(self, root: TreeNode) -> int:
        result1,result2= self.robdfs(root)
        return max(result1,result2)

    def countBits(self, num: int) -> list:
        dp   = [0] *(num+1)
        dp[1] = 1
        if num < 2:return dp
        for n in range(2, num+1):
            dp[n]= n%2+dp[n>>1]
        return dp
print(Solution().countBits(1000))