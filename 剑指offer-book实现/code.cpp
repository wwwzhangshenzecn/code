/*
此为剑指offer第二版 C++ 题目解法
*/
#include<algorithm>
#include<iostream>
#include<vector>
#include<stack>
#include<string>
#include<iterator>
#include<set>
using namespace std;

//T3
set<int> duplicateNumber(const vector<int>& arr) {
	//找出数组中重复的数字
	// all(arr) < arr.size
	vector<bool> dict(arr.size(), false);
	set<int> results;

	for each (const int n in arr)
	{
		if (n >= arr.size()) continue;
		if (dict[n])
			results.insert(n);
		else
			dict[n] = true;
	}

	return results;
}

//T4
/*
在一个二维数组中，每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序
再次二维数组中，查找整数的位置.
*/
pair<int, int> FindInTwoDemension(const vector<vector<int>>& arr, int key) {
	if (arr.size() == 0) return{ -1,-1 };
	const int ROWMAX = arr.size(), COLMAX = arr[0].size();
	int row = 0, column = COLMAX - 1;
	while (row < ROWMAX && column >= 0) {
		if (arr[row][column] == key)
			return{ row, column };
		else {
			if (arr[row][column] < key)
				row++;
			else
				column--;
		}
	}
	return{ -1, -1 };
}

//T5
/*
将字符串里的空格替换成 %20
*/
void replaceSpace(string& s, const string& re = "%20") {
	for (int i = 0; i < s.size(); i++) {
		if (s[i] == ' ')
			s.replace(i, 1, re);
	}
}

struct LinkNode
{
	int value;
	LinkNode* next;
	LinkNode(int value) :value(value), next(nullptr) {}
};

//T5
/*
反向打印链表
*/
void PrintReversingLint(LinkNode* head) {
	if (head != nullptr) {
		stack<decltype(head->value)> sk;
		while (head != nullptr) {
			sk.push(head->value);
			head = head->next;
		}
		while (!sk.empty()) {
			cout << sk.top() << endl;
			sk.pop();
		}
	}
}

//T7  前序中序建树
class BinaryTreeNode
{
public:
	int value;
	BinaryTreeNode* left, *right;
	BinaryTreeNode(int value,
		BinaryTreeNode* left = nullptr, BinaryTreeNode* right = nullptr) :value(value), left(left), right(right) {}
	~BinaryTreeNode() {}
};

BinaryTreeNode* BuildTreePreAndMid(vector<int>& pre, vector<int>& mid) {
	if (pre.size() == 0 || mid.size() == 0) return nullptr;
	BinaryTreeNode* root = new BinaryTreeNode(pre[0]);
	auto indexIter = find(mid.begin(), mid.end(), root->value);
	if (indexIter == mid.end())
		return nullptr;
	pre.erase(pre.begin());
	root->left = BuildTreePreAndMid(pre, vector<int>(mid.begin(), indexIter));
	root->right = BuildTreePreAndMid(pre, vector<int>(indexIter + 1, mid.end()));
	return root;
}

//T8
/*
给定以可二叉树和其中一个节点,如何找出中序遍历的下一个节点?
树中节点除了有两个分别指向左右节点的指针,还有一个指向父节点的指针.
*/
class BinaryTreeNodeF {
public:
	int value;
	BinaryTreeNodeF* left, *right, *father;
	BinaryTreeNodeF(int value,
		BinaryTreeNodeF* left = nullptr, BinaryTreeNodeF* right = nullptr,
		BinaryTreeNodeF* father = nullptr) :value(value), left(left), right(right), father(father) {}
	~BinaryTreeNodeF() {}
};

BinaryTreeNodeF* FindNextOfPreOrder(BinaryTreeNodeF* node) {
	if (node == nullptr)
		return nullptr;
	if (node->right != nullptr) { // 右子树,右子树的最左节点
		node = node->right;
		while (node->left != nullptr)
			node = node->left;
		return node;
	}
	//为叶节点,则下一节点为第一个有右子树的祖先节点
	while (node->right == nullptr) {
		if (node->father == nullptr) // 没有右子树的根节点，下一节点为根
			break;
		node = node->father;
	}
	return node;
}

//T9
/*
两个栈实现一个队列
*/
template<typename T>
class CQueue {
public:
	CQueue() {}
	void push(const T& t);
	const T top();
	void pop();
	void Print();
private:
	stack<T> store;
};

template<typename T>
void CQueue<T>::push(const T& t) {
	store.push(t);
}

template<typename T>
const T CQueue<T>::top() {
	stack<T> temp;
	T result;
	while (!store.empty()) {
		temp.push(store.top());
		store.pop();
	}
	result = temp.top();
	while (!temp.empty()) {
		store.push(temp.top());
		temp.pop();
	}
	return result;
}

template<typename T>
void CQueue<T>::pop() {
	stack<T> temp;
	while (!store.empty()) {
		temp.push(store.top());
		store.pop();
	}
	temp.pop();
	while (!temp.empty()) {
		store.push(temp.top());
		temp.pop();
	}
}

template<typename T>
void CQueue<T>::Print() {
	stack<T> temp;
	while (!store.empty()) {
		temp.push(store.top());
		store.pop();
	}
	while (!temp.empty()) {
		cout << temp.top() << " ";
		store.push(temp.top());
		temp.pop();
	}
	cout << endl;
}

//T10
/*
斐波拉契数列
*/

int Fibonacci(int n) {
	if (n <= 0)
		return 0;
	if (n == 1 || n == 2) return 1;
	int f1 = 1, f2 = 1, f3;
	while (n-- >= 3) {
		f3 = f1 + f2;
		f1 = f2;
		f2 = f3;
	}
	return f2;
}
//T11
/*
把一个数组最开始的若干个元素搬到数组的末尾,我们称之为数组的旋转.
输入递增排序的数组的一个旋转,输出旋转数组的最小元素.
*/
int Minrotate(vector<int> arr) {
	if (arr.size() == 0) return -1;
	int result = arr[0];
	for (int i = 1; i < arr.size(); i++) {
		if (arr[i] < arr[i - 1]) {
			result = arr[i];
			break;
		}
	}
	return result;
}

//T12
/*
//回溯法
*/

bool findPath(const vector<vector<char>>& matrix, vector<vector<bool>>& visited, const string& word, int x, int y) {
	if (word.size() == 0)
		return true;
	if (x < 0 || y < 0 || x >= matrix.size() || y >= matrix[0].size() || visited[x][y] == false || matrix[x][y] != word[0])
		return false;
	bool hasPath;
	visited[x][y] = false;
	hasPath = findPath(matrix, visited, word.substr(1, word.size()), x, y - 1) ||
		findPath(matrix, visited, word.substr(1, word.size()), x, y + 1) ||
		findPath(matrix, visited, word.substr(1, word.size()), x + 1, y) ||
		findPath(matrix, visited, word.substr(1, word.size()), x - 1, y);
	visited[x][y] = true;
	return hasPath;
}

bool hasPathCore(vector<vector<char>>& matrix, const string& word) {
	if (word.size() == 0 || matrix.size() == 0 || matrix[0].size() == 0)
		return false;
	vector<vector<bool>> visited(matrix.size(), vector<bool>(matrix[0].size(), true));
	for (int i = 0; i < matrix.size(); i++) {
		for (int j = 0; j < matrix[i].size(); j++) {
			if (findPath(matrix, visited, word, i, j))
				return true;
		}
	}
	return false;
}

//T13
bool check(pair<int, int> coor, int maxK) {
	string num = to_string(coor.first) + to_string(coor.second);
	int sum = 0;
	for each(const char ch in num) {
		sum += ch - '0';
	}
	return sum > maxK;
}

int movingCountOfRobot(pair<int, int>& coor, int maxK, vector<vector<bool>>& visited = vector<vector<bool>>{}, pair<int, int> start = { 0,0 }) {
	if (start.first < 0 || start.second < 0 || start.first >= coor.first ||
		start.second >= coor.second || check({ start.first, start.second }, maxK))
		return 0;
	if (visited.size() == 0)
		visited = vector<vector<bool>>(coor.first, vector<bool>(coor.second, true));
	if (visited[start.first][start.second] == false)
		return 0;
	int stepCount = 0;
	visited[start.first][start.second] = false;
	return 1 + movingCountOfRobot(coor, maxK, visited, { start.first, start.second + 1 })
		+ movingCountOfRobot(coor, maxK, visited, { start.first + 1, start.second });
}

//T14
int maxProduct(int n) {
	if (n <= 0) return 0;
	if (n <= 3) return n;
	int m = 0;
	for (int i = 3; i < n; i++)
		m = max(m, maxProduct(i)*maxProduct(n - i));
	return m;
}
int maxProductAfterCutting(int length) {
	if (length < 2)return 0;
	if (length == 2) return 1;
	if (length == 3) return 2;
	return maxProduct(length);
}

//T15
int NumberOne(int n) {
	int count = 0;
	while (n) {
		++count;
		n = (n - 1)&n;
	}
	return count;
}

int NumberOne1(int num) {
	int count = 0;
	while (num) {
		if (num & 1)
			count++;
		num = num >> 1;
	}
	return count;
}












//int main() {
//	int result = atoi("12345");
//	cout << result << endl;
//
//	//copy(results.begin(), results.end(), ostream_iterator<int>(cout, ""));
//	return 0;
//}