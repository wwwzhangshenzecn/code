import heapq
import time


class CompareList:
    def __init__(self, l):
        self.list = l

    def __lt__(self, other):
        if self.list[0] < other.list[0]:
            return True
        else:
            return False


#
#
# class Solution:
#
#     def com(self, a, b):
#         if a[0] < b[0]:
#             return True
#
#     def lexicalOrder1(self, n: int) -> list:
#         result = []
#         for i in range(1, n + 1):
#             result.append([str(i), i])
#         result.sort(key=lambda x: x[0])
#         return [i[1] for i in result]
#
#     def lexicalOrder(self, n: int) -> list:
#         result = []
#         for i in range(1, n + 1):
#             result.append(CompareList([str(i), i]))
#         heapq.heapify(result)
#         r = []
#         while result:
#             r.append(heapq.heappop(result).list[1])
#         return r
#

from turtle import *


def onclick(x, y):
    print(x, y)


def FiveStart():
    pen = Pen()
    pen.pensize(5)
    pen.color('red')
    pen.forward(200)
    for i in range(4):
        pen.right(144)
        pen.forward(200)

    done()


def square():
    pen = Pen()
    pen.speed(0)
    i = 1
    while i <= 100:
        pen.forward(i)
        pen.left(90)
        i += 1

    done()


def sun():
    pen = Pen()
    pen.color('red', 'yellow')
    pen.begin_fill()
    pen.speed(0)
    pen.hideturtle()

    while 1:
        pen.forward(200)
        pen.left(170)
        if (abs(pen.pos()) < 1):
            pen.end_fill()
            break

    done()


def HL():
    pen = Pen()
    pen.color('red', 'blue')
    pen.begin_fill()
    pen.speed(0)
    pen.hideturtle()

    for i in range(200, 0, -1):
        pen.lt(-5)
        pen.circle(i)
        if (abs(pen.pos()) < 1):
            pen.end_fill()
            pen.begin_fill()

    done()


def Yuan():
    pen = Pen()
    pen.color('red', 'blue')
    pen.speed(0)
    pen.hideturtle()

    for i in range(200, 0, -1):
        pen.rt(i * 2)
        pen.forward(1)
        pen.circle(i)
        time.sleep(0.05)
        pen.clear()

    done()


import collections


def lastRemaining(n: int) -> int:
    dict = collections.defaultdict(int)
    for i in range(1, 5):
        dict[i] = get(i, {})

    for i in range(5, n + 1):
        dict[i] = get(i, dict)
    print(dict)
    return None


def get(n: int, dict: dict):
    List = list(range(1, n + 1))
    step = 2

    while len(List) > 1:
        if len(List) < len(dict):
            return List[[len(List)]] - 1
        R = (1, len(List), step) if step > 0 else (len(List) - 2, -1, step)
        List = [List[i] for i in range(*R)]
        if step < 0: List.reverse()
        step *= -1

    return List[0] - 1


# from proj import tasks
from multiprocessing import Process, Queue

que = Queue()
result = Queue()
num = 2


def worker(que: Queue, result: Queue) -> int:
    while 1:
        A = que.get()
        if A == 'end': break
        n = len(A)
        temp = 0
        for i in range(n):
            temp += i * A[i]
        result.put(temp)


def product(que: Queue, A):
    for i in range(len(A)):
        A = A[1:] + [A[0]]
        que.put(A)
    for _ in range(num):
        que.put('end')


class Solution:

    def findR(self, path: str, delt: str, reg: str) -> None:
        print(path)
        list = delt.split(reg)
        fatherdir = list[0]

        for delt in list[1:]:
            self.findR(path + fatherdir, delt, reg + r'\t')

    def lengthLongestPath(self, input: str) -> int:
        path = ''
        startReg = r'\n\t'
        self.findR(path, input, startReg)

    def isRectangleCover(self, rectangles) -> bool:
        '''
        通过面积的拼合来计算，不过这种方式只能计算非重合矩形
        :param rectangles:
        :return:
        '''
        hp = set()
        area = 0
        for i in rectangles:
            lb = [i[0], i[1]]
            lt = [i[0], i[3]]
            rt = [i[2], i[3]]
            rb = [i[2], i[1]]
            area += (i[2] - i[0]) * (i[3] - i[1])
            for i in [lb, lt, rt, rb]:
                if i not in hp:
                    hp.add(i)
                else:
                    hp.remove(i)
        if len(hp) != 4:
            return False
        hp = sorted(hp)
        first, last = hp[0], hp[-1]
        return area == (last[0] - first[0]) * (last[1] - first[1])

    def decodeString(self, s: str) -> str:
        start, end = 0, 'n'
        result = ''
        stack = []
        for i in range(len(s)):
            if (end == 'n'):
                if '0' <= s[i] <= '9':
                    end = ']'
                    result += s[start:i]
                    start = i
            elif (end == ']'):
                if s[i] == '[':
                    stack.append(s[i])
                elif s[i] == ']':
                    stack.pop()
                    if stack.__len__() == 0:
                        ls = s[start:i].find('[') + start
                        result += self.decodeString(s[ls + 1:i]) * int(s[start:ls])
                        start = i + 1
                        end = 'n'
        result += s[start:]
        return result

    # 395.
    def longestSubstring(self, s: str, k: int) -> int:
        AlphabetIndex = collections.defaultdict(int)
        AlphabetLocal = collections.defaultdict(list)
        result = 0
        ltA = set(s)
        for i in range(len(s)):
            AlphabetLocal[s[i]].append(i)
            AlphabetIndex[s[i]] += 1
            if AlphabetIndex[s[i]] >= k and s[i] in ltA:
                ltA.remove(s[i])
        index = [-1] + sorted([i for c in ltA for i in AlphabetLocal[c]]) + [s.__len__()]
        if len(index) == 2:
            return len(s)
        for i in range(len(index) - 1):
            result = max(self.longestSubstring(s[index[i] + 1:index[i + 1]], k), result)
        return result

    # 396.
    def maxRotateFunction(self, A: list) -> int:
        total = sum(A)
        n = len(A)
        if n <= 1:
            return 0
        cur = 0
        for i in range(n):
            cur += i * A[i]
        maxv = cur
        for i in range(1, n):
            cur += total - A[-i] - (n - 1) * A[-i]
            maxv = max(maxv, cur)
        return maxv

    def integerReplacement1(self, n: int) -> int:
        if n <= 1: return 0
        if n == 2: return 1
        if n == 3: return 2
        tables = [0] * (n + 3) + [2 ** 32]
        tables[1], tables[2], tables[3] = 0, 1, 2
        if n <= 3:
            return tables[n]
        for i in range(4, n + 3, 2):
            tables[i] = tables[i // 2] + 1

            tables[i - 1] = min(tables[i - 2], tables[i]) + 1

        return tables[n]

    def integerReplacement(self, n: int) -> int:
        if n == 1: return 0
        if n == 2: return 1
        if n %2 == 1:
            return min(self.integerReplacement(n+1),
                       self.integerReplacement(n-1))+1
        else:
            return self.integerReplacement(n//2)+1

    def FindcalcBFS(self, edges, weight, x, y):
        stack = [x]
        path = set([x])
        flag =True

        while stack!= [] and flag:
            node = stack.pop()
            for n in edges[node]:
                if n not in path:
                    if (x, n) not in weight.keys():
                        weight[(x, n)] = [weight[(x, node)][0]*weight[(node, n)][0],
                                          weight[(x, node)][1] * weight[(node, n)][1]]
                    stack.append(n)
                    path.add(n)
                if n == y:
                    return weight[(x, y)][0] / weight[(x, y)][1]
        return -1.0


    # def calcEquation(self, equations: List[List[str]],
    #  values: List[float],
    #  queries: List[List[str]]) -> List[float]:
    def calcEquation(self, equations, values, queries):
        weight = collections.defaultdict(int)
        edges = collections.defaultdict(list)
        for i , (x, y) in enumerate(equations):
            weight[(x, y)] = [values[i], 1]
            weight[(y, x)] = [1, values[i]]
            edges[x].append(y)
            edges[y].append(x)

        results = []
        for x, y in queries:
            if x not in edges.keys() or y not in edges.keys():
                results.append(-1.0)
            elif x == y:
                results.append(1.0)
            else:
                results.append(self.FindcalcBFS(edges, weight, x, y))
        return results

if __name__ == '__main__':
    print(Solution().calcEquation([ ["a", "b"], ["b", "c"] ],
                                  [2.0, 3.0],
                                  [["a", "c"], ["b", "a"], ["a", "e"], ["a", "a"], ["x", "x"]]))

    # t= time.time()
    # result = []
    # for i in range(5):
    #     result.append(tasks.add.delay(1,1))
    #
    # print([re.get() for re in result])
    # print(time.time()-t)
