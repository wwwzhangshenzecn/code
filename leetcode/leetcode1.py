#coding:utf-8
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Node:
    def __init__(self, val, left=None, right=None, next=None):
        self.val = val
        self.left = left
        self.right = right
        self.next = next


class Solution:

    def prebuildTree(self, preorder: list, inorder: list) -> TreeNode:
        if len(inorder) == 0:
            return None
        else:
            index = inorder.index(preorder[0])
            root = TreeNode(preorder[0])
            del preorder[0]
            root.left = self.buildTree(preorder,inorder[0:index])
            root.right = self.buildTree(preorder,inorder[index+1:])
            return root

    def posbuildTree(self, inorder, postorder) -> TreeNode:
        if len(inorder)==0:
            return None
        else:
            index = inorder.index(postorder[-1])
            root = TreeNode(inorder[index])
            del postorder[-1]
            root.right = self.buildTree(inorder[index+1:], postorder)
            root.left = self.buildTree(inorder[0:index], postorder)
            return root

    def sortedArrayToBST(self, nums: list) -> TreeNode:
        if len(nums) == 0:return None
        elif len(nums) == 1:return TreeNode(nums[0])
        else:
            root = TreeNode(nums[len(nums)//2])
            root.left = buildTree(nums[:len(nums)//2])
            root.right = buildTree(nums[len(nums)//2+1:])
            return root

    def isBalanced(self, root: TreeNode) -> bool:
        def height(root):
            if root == None:
                return 0
            else:
                return max(height(root.left),height(root.right))+1
        def f(root):
            if not root:
                return -1
            else:
                left = height(root.left)
                if left != 0 and left == False: return False
                right = height(root.right)
                if left != 0 and right == False:return False
                if abs(left - right) >= 2 :return False
                return left-right

        return True if abs(f(root))<=1 else False


    def minDepth(root: TreeNode) -> int:
        if not root:return 0
        p ,q = [],[root]
        result = [[]]
        while p:
            root = p[0]
            del p[0]
            if root.left:q.append(root.left)
            if root.right:q.append(root.right)
            if not root.left and not root.right:
                return len(result)
            result[-1].append(root.val)
            if not p:
                p,q = q.p
                result.append([])


    def hasPathSum(self, root: TreeNode, sum: int) -> bool:
        if not root:return False
        def f(root ,sum):
            if root is None:return False
            else:
                sum -= root.val
                if sum == 0 and not root.left and not root.right:
                    return True
                return f(root.left,sum) or f(root.right,sum)
            
        return True if f(root,sum) else False

    def pathSum(self, root: TreeNode, sum: int) -> list:
        results = []
        def f(root, sum, result=None):
            if root is None:pass
            else:
                sum -= root.val
                if sum == 0 and not root.left and not root.right:
                    results.append(result+[root.val])
                f(root.left, sum, result+[root.val]) 
                f(root.right, sum, result+[root.val])
        f(root,sum,[])
        return results
        
        
    def flatten(self, root: TreeNode) -> None:
        """
        Do not return anything, modify root in-place instead.
        """
        if not root:return root
        stack = []
        def preorder(root:TreeNode):
            if not root:
                pass
            else:
                stack.append(root)
                preorder(root.left)
                preorder(root.right)
        preorder(root)
        if len(stack) > 1:
            for i, _ in enumerate( stack[:-1]):
                stack[i].right = stack[i+1]
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
                temp[i] = sum(table[i+1:])
            table = temp
        return sum(table)


    def generate(self, numRows: int) -> list:
        result = [[1]]
        if numRows == 0:return []
        if numRows == 1:return [[1]]
        for i in range(1,numRows):
            result.append([])
            for j in range(i+1):
                if j==0:
                    result[-1].append(result[-2][0])
                elif j==i:
                    result[-1].append(result[-2][-1])
                else:
                    result[-1].append(result[-2][j] +result[-2][j-1])
        return result


    def getRow(self, numRows: int) -> list:
        result = [1]
        if numRows == 0:return [1]
        for i in range(1,numRows+1):
            temp = []
            for j in range(i+1):
                if j==0:
                    temp.append(result[0])
                elif j==i:
                    temp.append(result[-1])
                else:
                    temp.append(result[j] +result[j-1])
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
                for i in range(len(q)-1):
                    q[i].next = q[i+1]
                p, q = q, p

    def getMinimumDifference(self, root: TreeNode) -> int:
        # 530.
        if not root:return 0
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
        for i in range(1,len(inorder)):
            if inorder[i] - inorder[i-1] < MinD:
                MinD =inorder[i] - inorder[i-1]
        return MinD


    def checkSubarraySum(self, nums: list, k: int) -> bool:
        # 523.
        if len(nums) == 1:
            return False
        if k == 0:
            for i in range(len(nums)-1):
                if nums[i] ==0 and nums[i+1]==0:
                    return True
            return False
        
        for i in range(len(nums)):
            table = [0] *(i+1)
            table[-1]=nums[i]
            for j in range(i-1,-1,-1):
                table[j] = table[j+1] + nums[j]
                if table[j] % k == 0:return True
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
        if len(triangle) == 0:return 0
        table = triangle[-1]
        for i in range(-2,-len(triangle)-1,-1):
            for j in range(len(triangle[i])):
                table[j] = min(table[j], table[j+1]) + triangle[i][j]
        
        return table[0]
        
    def maxProfit(self, prices: list) -> int:
        Maxp = 0
        if len(prices)<=1:return 0
        Minp = prices[0]
        for i in range(1,len(prices)):
            Minp = min(Minp, prices[i-1])
            Maxp = max(Maxp, prices[i]-Minp)
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


    def maxProfit3(self, prices:list) -> int:
        # 局部最高-循环
        i = 0
        while i < len(prices):
            if i >= 1:
                if prices[i] == prices[i-1]:
                    del prices[i]
                    continue
            i += 1

        local = [i for i in range(1, len(prices)-1) if prices[i+1] < prices[i] and prices[i] >  prices[i-1]]
        local.append(len(prices)-1)
        twice_max = 0
        for i in local:
            twice_max = max(twice_max, self.maxProfit(prices[i+1:]) + self.maxProfit(prices[0:i+1]))

        return twice_max

    def isPalindrome(self, s: str) -> bool:
        s = ''.join([c for c in s if c.isalpha() or c.isdigit()]).lower()
        return s == s[::-1]


def A():
    pass