/*
此为剑指offer第二版 C++ 题目解法,
算法中可能会存在逻辑/边界错误，请自己修正。
*/
#include<numeric>
#include<algorithm>
#include<iostream>
#include<vector>
#include<stack>
#include<string>
#include<iterator>
#include<set>
#include<atomic>
#include<mutex>

using namespace std;

////T2
//class singleton {
//private:
//	singleton();
//	singleton(singleton* s) {}
//public:
//	static singleton* getInstance();
//	static singleton* instance;
//
//};
//
//static singleton* singleton::instance = nullptr;
//
//std::atomic<singleton*> singleton::instance;
//std::mutex singleton::mutex;
//
//
//singleton* singleton::getInstance() {
//	singleton* temp = instance.load(std::memory_order_relaxed);
//	std::_Atomic_thread_fence(std::memory_order_acquire);
//	if (temp == nullptr) {
//		std::lock_guard<std::mutex> lock(mutex);
//		temp = instance.load(std::memory_order_relaxed);
//		if (temp == nullptr) {
//			temp = new singleton;
//			std::_Atomic_thread_fence(std::memory_order_release);
//			instance.store(temp, std::memory_order_relaxed);
//		}
//	}
//	return temp;
//}


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


//T16
double PowerWithUnsignedExponent(double base, int exponent) {
	if (exponent == 0)
		return 1;
	if (exponent == 1)
		return base;
	double result = PowerWithUnsignedExponent(base, exponent >> 1);
	result *= result;
	if (exponent & 1)
		result *= base;
	return result;
}

const double SuperPower(double base, int exponent) {
	if (base == 0.0) return 0;
	bool flag;
	if (exponent >= 0) {
		flag = true;
	}
	else {
		flag = false;
		exponent *= -1;
	}

	double result = PowerWithUnsignedExponent(base, exponent);
	if (flag)
		return result;
	else
		return 1.0 / result;
}

//T17

void PrintDigitsRecuse(int deep, vector<int>& num, int result = 0) {
	if (deep != 0) {
		for (int i = 0; i < num.size(); i++)
			PrintDigitsRecuse(deep - 1, num, result * 10 + num[i]);
	}
	else {
		cout << result << endl;
	}
}

void PrintToMaxofDigits(int n) {
	vector<int> num(10, 0);
	iota(num.begin(), num.end(), 0);
	PrintDigitsRecuse(n, num);
}

//T18
void DeleteNode(LinkNode*  pListHead, LinkNode* pToBeDel) {
	if (pListHead == nullptr || pToBeDel == nullptr)
		return;
	if (pListHead == pToBeDel) {
		delete pListHead;
		pListHead = nullptr;
	}

	if (pToBeDel->next != nullptr) {
		//node 不是尾节点，如果是尾节点，必须从头开始查找
		LinkNode* pNext = pToBeDel->next;
		pToBeDel->value = pNext->value;
		pToBeDel->next = pNext->next;
		delete pNext;
	}
	else {
		// 尾节点，从头查找
		LinkNode* pre = pListHead;
		while (pre->next != pToBeDel)
			pre = pre->next;
		pre->next == nullptr;
		delete pToBeDel;
		pToBeDel = nullptr;
	}
}


//T19
// DFA / NDFA
bool mathCoreDFA(string& pattern, string& str) {
	if (str.size() == 0 && pattern.size() == 0)
		return true;
	if (str.size() != 0 && pattern.size() == 0)
		return false;
	if (pattern.size() == 1) {
		if (pattern == "." && str.size() == 1)
			return true;
		else
			return pattern == str;
	}
	if (pattern[1] == '*')
		if (str.size() == 0)
			return mathCoreDFA(pattern.substr(2, pattern.size()), str);
	if (pattern[0] == str[0])
		return mathCoreDFA(pattern.substr(2, pattern.size()), str.substr(1, str.size())) ||
		mathCoreDFA(pattern, str.substr(1, str.size()));
	else
		return mathCoreDFA(pattern.substr(2, pattern.size()), str);
	if (pattern[0] == str[0] || pattern[0] == '.')
		return mathCoreDFA(pattern.substr(1, pattern.size()), str.substr(1, str.size()));
	return false;
}

bool match(string pattern, string str) {
	if (pattern == str)
		return true;

	if (pattern.size() == 0)
		return false;

	return mathCoreDFA(pattern, str);
}

//T20

bool ScanUnsignedInterger(const string& str, int& pos) {
	const int prepos = pos;
	while (pos < str.size() && str[pos] >= '0' && str[pos] <= '9')
		++pos;
	return pos > prepos;
}

bool ScanInterger(const string& str, int &pos) {
	if (str[pos] == '+' || str[pos] == '-')
		++pos;
	return ScanUnsignedInterger(str, pos);
}

bool IsNumeric(string str) {
	if (str.size() == 0)
		return false;
	int pos = 0;
	bool numeric = ScanInterger(str, pos);
	if (str[pos] == '.') {
		numeric = numeric || ScanUnsignedInterger(str, ++pos);
	}
	if (str[pos] == 'e' || str[pos] == 'E') {
		numeric = numeric || ScanInterger(str, ++pos);
	}
	return numeric;
}

//T21
bool Judge(int index) { // 判断是否是奇数
	if (index & 1)
		return true;
	else
		return false;
}

void ReorderArr(vector<int>& arr, bool(*Judge)(int)) {
	int left = 0, right = arr.size() - 1;
	while (left < right) {
		while (Judge(arr[left]))
			++left;
		while (!Judge(arr[right]))
			++right;
		if (left < right)
			swap(arr.begin() + left, arr.begin() + right);
	}
}

//T22
LinkNode* FindKNode(LinkNode* pLinkHead, unsigned int k) {
	if (pLinkHead == nullptr)
		return nullptr;
	LinkNode* p1 = pLinkHead;
	while (--k) {
		if (p1->next == nullptr) // 链长长度小于k
			return nullptr;
		p1 = p1->next;
	}
	LinkNode* p2 = pLinkHead;
	while (p1 != nullptr) {
		p2 = p2->next;
		p1 = p1->next;
	}
	return p2;
}

//T23

LinkNode* MeetingNode(LinkNode* pLinkHead) {
	LinkNode* pfast = pLinkHead->next;
	LinkNode* pslow = pLinkHead;
	while (pfast != pslow) {
		if (pfast == nullptr)
			return pfast;

		pslow = pslow->next;
		pfast = pfast->next;
		if (pfast != nullptr)
			pfast = pfast->next;
	}
}


LinkNode* EntryNodeOfLoop(LinkNode* pLinkHead) {
	if (pLinkHead == nullptr) return nullptr;
	LinkNode* meetNode = MeetingNode(pLinkHead);
	if (meetNode == nullptr) // 无环
		return nullptr;

	int count = 1;
	const LinkNode* local = meetNode;
	LinkNode* CNode = meetNode;
	while (CNode->next != local) {
		++count;
		CNode = CNode->next;
	}
	LinkNode* p1 = pLinkHead;
	while (count--) {
		p1 = p1->next;
	}

	LinkNode* p2 = pLinkHead;
	while (p2 != p1) {
		p2 = p2->next;
		p1 = p1->next;
	}
	return p2;
}

//T24
LinkNode* ReverseLink(LinkNode* pLinkhead) {
	if (pLinkhead == nullptr) return nullptr;
	if (pLinkhead->next == nullptr) return pLinkhead;

	LinkNode* pre = pLinkhead;
	LinkNode* pos = pLinkhead->next;
	while (pos != nullptr) {
		if (pre == pLinkhead)
			pre->next == nullptr;
		LinkNode* next = pos->next;
		pos->next = pre;
		pre = pos;
		pos = next;
	}
	pLinkhead = pre;
	return pLinkhead;
}

//T25
LinkNode* Merge(LinkNode* pHead1, LinkNode* pHead2) {
	LinkNode* pMergeHead = new LinkNode(0);
	LinkNode* p1 = pHead1;
	LinkNode* p2 = pHead2;
	LinkNode* r = pMergeHead;
	while (p1 != nullptr || p2 != nullptr) {
		if (p1->value < p2->value) {
			r->next = new LinkNode(p1->value);
			r = r->next;
			p1 = p1->next;
		}
		else {
			r->next = new LinkNode(p2->value);
			r = r->next;
			p2 = p2->next;
		}
	}
	if (p1 == nullptr)
		p1 = p2;
	while (p1 != nullptr) {
		r->next = new LinkNode(p1->value);
		r = r->next;
		p1 = p1->next;
	}
	return pMergeHead;
}

//T26

bool DoesTree1HasTree2(BinaryTreeNode* root1, BinaryTreeNode* root2) {
	if (root2 == nullptr) return true;
	if (root1 == nullptr) return false;
	if (root1->value != root2->value)
		return false;
	return DoesTree1HasTree2(root1->left, root2->right) || DoesTree1HasTree2(root1->right, root2->right);
}

bool HasSubTree(BinaryTreeNode* root1, BinaryTreeNode* root2) {
	if (root1 == nullptr || root2 == nullptr) return false;
	if (root1->value == root2->value) {
		if (DoesTree1HasTree2(root1, root2))
			return true;
		return HasSubTree(root1->left, root2) || HasSubTree(root1->right, root2);
	}
}



int main() {
	auto result = IsNumeric("+12.3e+5");
	cout << result << endl;
	//copy(results.begin(), results.end(), ostream_iterator<int>(cout, ""));
	return 0;
}