#include<set>
#include<deque>
#include<bitset>
#include<queue>
#include<iterator>
#include<memory>
#include<assert.h>
#include<memory>
#include<string>
#include<iostream>
#include<time.h>
#include<map>
#include<vector>
#include<algorithm>
#include<functional>
#include<math.h>
#include<numeric>
using namespace std;

class A {
private:
	int x, y;
public:
	A() :x(0), y(0) {}
	void setx(int x) { this->x = x; }
	void sety(int y) { this->y = y; }
	friend int get(A);
};



int get(A a) {
	return a.x + a.y;
}
////Definition for singly-linked list.
//struct ListNode {
//	int val;
//	ListNode *next;
//	ListNode(int x) : val(x), next(NULL) {}
//};


struct point3 {
	int a;
	int b;
	int c;
}Point3;

#define Point_print(pd) cout<<pd.a<<endl;

int func(int m, int n) {

	return (m > n) ? m : n;
}


/**
* Definition for singly-linked list.
* struct ListNode {
*     int val;
*     ListNode *next;
*     ListNode(int x) : val(x), next(NULL) {}
* };
*/
struct ListNode {
	int val;
	ListNode *next;
	ListNode(int x) : val(x), next(NULL) {}

};

ostream& operator<<(ostream& os, ListNode *node) {
	while (node != NULL) {
		os << node->val << " ";
		node = node->next;
	}
	return os;
}
struct TreeNode {
	int val;
	TreeNode *left;
	TreeNode *right;
	TreeNode(int x) : val(x), left(NULL), right(NULL) {}

};

class Solution2 {
public:


	//2.
	ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
		ListNode *result = new ListNode(0);
		ListNode * r = result;
		int flag = 0;
		while (l1 != NULL && l2 != NULL) {
			int temp = l1->val + l2->val + flag;
			ListNode * node = new ListNode(temp % 10);

			// 进位标志
			if (temp >= 10)
				flag = 1;
			else
				flag = 0;

			l1 = l1->next;
			l2 = l2->next;
			r->next = node;
			r = r->next;
		}
		if (l1 == NULL)
			l1 = l2;

		while (l1 != NULL) {
			int temp = flag + l1->val;
			ListNode *node = new ListNode(temp % 10);
			if (temp >= 10)
				flag = 1;
			else
				flag = 0;

			l1 = l1->next;
			r->next = node;
			r = r->next;
		}

		if (flag == 1) {
			ListNode * node = new ListNode(1);
			r->next = node;
		}
		r = result;
		result = result->next;
		delete r;
		return result;
	}

	//1.
	vector<int> twoSum(vector<int>& nums, int target) {
		vector<int> result;
		for (int i = 0; i < nums.size(); i++) {
			vector<int>::iterator iter = find(nums.begin() + i + 1, nums.end(), target - nums[i]);
			if (iter != nums.end()) {
				result.push_back(i),
					result.push_back(iter - nums.begin());
				break;
			}
		}
		return result;
	}


	//3.
	int lengthOfLongestSubstring(string s) {
		int i = 0, j = 0, ans = 0;
		map<char, int> dict;
		int n = s.size();
		// 由i 和j 组成的滑动窗口，j一直向后移动，并且将j位置的字母记录下来-->dict
		// 若是 j位置的字母，则记录最后出现的位置，避免中间重复
		while (i < n && j < n) {
			if (dict.count(s[j])>0)
				i = max(i, dict[s[j]]);
			dict[s[j]] = j + 1;
			ans = max(ans, j - i + 1);
			j++;
		}
		return ans;
	}


	//4.
	double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
		int size = nums1.size() + nums2.size();
		vector<int> b(size, 0);
		merge(nums1.begin(), nums1.end(), nums2.begin(), nums2.end(), b.begin());
		if (size % 2 == 0) {
			return (*(b.begin() + size / 2) + *(b.begin() + size / 2 + 1)) / 2.0;
		}
		else
			return *(b.begin() + (size + 1) / 2);

	}

	// 判断是否回文串
	bool Palindrome(const string& s) {
		int i = 0, j = s.size() - 1;
		while (i <= j) {
			if (s[i] != s[j])
				return false;
			i++, j--;
		}
		return true;
	}

	//5.
	string longestPalindrome(string s) {
		string  result;
		int size = s.size();
		if (size < 2 || Palindrome(s)) { return s; }
		int start = 0, maxlen = 0;
		for (int i = 0; i < size; i++) {
			if (i - maxlen - 1 >= 0 && Palindrome(
				s.substr(i - maxlen - 1, maxlen + 2))) {
				start = i - maxlen - 1;
				maxlen += 2;
				continue;
			}

			if (i - maxlen >= 0 && Palindrome(
				s.substr(i - maxlen, maxlen + 1))
				) {
				start = i - maxlen;
				maxlen++;
			}
		}

		return s.substr(start, maxlen);
	}

	//6.
	string convert(string s, int numRows) {
		// flag :向下移动 row++
		// flag ：向右上移动
		if (numRows <= 1)return s;
		int x = -1, y = -1;
		vector<vector<char>> sv(numRows, vector<char>(s.size(), ' '));
		int i = 0, j = 0;
		for (int k = 0; k < s.size(); k++) {
			sv[i][j] = s[k];

			if (i == numRows - 1 || i == -0) {
				x *= -1;
				y = (y + 1) % 2;
			}

			i += x;
			j += y;
		}
		string result;
		for (int i = 0; i < numRows; i++)
			for (int j = 0; j < s.size(); j++)
				if (sv[i][j] != ' ')
					result += sv[i][j];
		return result;
	}

	//7.
	long long reverse(long long x) {
		if (x > pow(2, 31) - 1 || x < -1 * pow(2, 31))
			return 0;
		int flag = x >= 0 ? 1 : -1;
		x = abs(x);
		string temp = to_string(x);
		std::reverse(temp.begin(), temp.end());
		long long result = 0;
		for (int i = 0; i < temp.size(); i++)
			result = result * 10 + temp[i] - '0';
		return result*flag;
	}
	int nthSuperUglyNumber1(int n, vector<int>& primes) {
		vector<int> list{ 1 };
		set<int> result;
		make_heap(list.begin(), list.end(), greater<int>());
		while (result.size() < n) {
			pop_heap(list.begin(), list.end(), greater<int>());
			int v = list.back();
			list.pop_back();

			for (int i = 0; i < primes.size(); i++) {
				int k = primes[i] * v;
				if (k > n * primes[primes.size() - 1]) continue;
				list.push_back(primes[i] * v);
				push_heap(list.begin(), list.end(), greater<int>());
			}

			result.insert(v);
		}

		return *max_element(result.begin(), result.end());
	}

	//310.
	typedef map<int, vector<int>> dict;
	vector<int> findMinHeightTrees1(int n, vector<vector<int>>& edges) {
		if (n == 0) return{};
		if (n == 1) return{ 0 };
		if (n == 2) return{ 0,1 };
		dict nodelist;
		for (int i = 0; i < n; i++)
			nodelist[i] = vector<int>();

		for (int i = 0; i < edges.size(); i++)
			nodelist[edges[i][0]].push_back(edges[i][1]),
			nodelist[edges[i][1]].push_back(edges[i][0]);

		vector<vector<int>> nodeh;
		int min_temp = n;
		for (int i = 0; i < n; i++) {
			nodeh.push_back({ i, 0 });
		}

		random_shuffle(nodeh.begin(), nodeh.end());

		for (int i = 0; i < n; i++) {
			cout << nodeh[i][0] << " " << nodelist[nodeh[i][0]].size() << "  ";
			if (nodelist[nodeh[i][0]].size() == 1)
				continue;
			nodeh[i][1] = DFS(nodeh[i][0], nodeh[i][0], 0, nodelist, min_temp);
			cout << " " << nodeh[i][1] << endl;
			min_temp = min(min_temp, nodeh[i][1]);
		}

		vector<int> result;


		for (int i = 0; i < nodeh.size(); i++) {
			if (nodeh[i][1] == min_temp)
				result.push_back(nodeh[i][0]);
		}

		return result;


	}

	int DFS(int start, int pre, int deep, dict& nodelist, int& temp_min) {
		if (deep > temp_min) return deep + 1;

		if (nodelist[start].size() == 1 && nodelist[start][0] == pre) {
			//遇到末端结点
			return 0;
		}
		else {
			auto list = nodelist[start];
			int m = 0; // 目前最小
			for (int i = 0; i < list.size(); i++) {
				//dfs
				if (list[i] != pre) {
					m = max(m, 1 + DFS(list[i], start,
						deep + 1, nodelist, temp_min));
				}
				if (m > temp_min) {
					return m;
				}
			}

			return m;
		}
	}

	//310.
	//叶子剪切法
	vector<int> findMinHeightTrees(int n, vector<vector<int>>& edges) {
		/*
			对叶子进行收缩，每一次都会遍历一遍。相当于同时减少一层，
			直到只剩下最后的根
			就好比一个四散的点，看成一个大圆。这个圆从外向内 逐渐收缩一层，最后收缩中心肯定是圆点

		*/

		if (n == 1) {
			return{ 0 };
		}
		vector<vector<int>> nbrs(n, vector<int>());
		for (int i = 0; i < edges.size(); i++) {
			nbrs[edges[i][0]].push_back(edges[i][1]),
				nbrs[edges[i][1]].push_back(edges[i][0]);
		}

		vector<int> cut, newcut;
		for (int i = 0; i < nbrs.size(); i++)
			if (nbrs[i].size() == 1)
				cut.push_back(i);

		while (cut.size() > 0) {

			for (int left = 0; left < cut.size(); left++) {
				for (int nb = 0; nb < nbrs[cut[left]].size(); nb++)
				{
					auto iter = find(nbrs[nbrs[cut[left]][nb]].begin(), nbrs[nbrs[cut[left]][nb]].end(), cut[left]);
					if (iter != nbrs[nbrs[cut[left]][nb]].end()) {
						nbrs[nbrs[cut[left]][nb]].erase(iter);
					}

					if (nbrs[nbrs[cut[left]][nb]].size() == 1)
						newcut.push_back(nbrs[cut[left]][nb]);
				}
			}

			if (newcut.size() == 0) break;
			cut = newcut;
			newcut = {};

		}
		return cut;

	}
	int nthSuperUglyNumber(int n, vector<int>& primes) {
		// 相对于 nthSuperUglyNumber1 将n反向计数
		// python会快很多
		vector<int> ugly{ 1 };
		set<int> result{ 1 };
		make_heap(ugly.begin(), ugly.end(), greater<int>());

		int un = 0;
		while (n > 1) {
			pop_heap(ugly.begin(), ugly.end(), greater<int>());
			un = ugly.back();
			ugly.pop_back();

			for (int i = 0; i < primes.size(); i++)
			{
				int p = primes[i];
				long long  item = p * un;
				if (item > pow(2, 32) - 1)
					continue;
				if (result.count(item) == 0) {
					ugly.push_back(item);
					push_heap(ugly.begin(), ugly.end(), greater<int>());
					result.insert(item);
				}
			}
			n--;
		}

		return ugly[0];

	}



	vector<int> countSmaller1(vector<int>& nums) {
		vector<int> result(nums.size(), 0);
		for (auto iter = nums.begin(); iter != nums.end(); iter++)
			result[iter - nums.begin()] = count_if(iter + 1, nums.end(), bind2nd(less<int>(), *iter));
		return result;
	}

	vector<int> countSmaller2(vector<int>& nums) {
		vector<int> result(nums.size(), 0);
		map<int, int> dict;

		for (int i = nums.size() - 1; i >= 0; i--) {
			for (auto iter = dict.begin(); iter != dict.end(); iter++) {
				if (iter->first < nums[i]) {
					result[i] += iter->second;
				}
			}

			if (dict.count(nums[i])>0) {
				dict[nums[i]]++;
			}
			else {
				dict.insert(make_pair(nums[i], 1));
			}

		}
		return result;
	}

	//315.
	vector<int> countSmaller3(vector<int>& nums) {
		// 有序插入
		vector<int> sonums;
		vector<int> result;

		std::reverse(nums.begin(), nums.end());
		for (int i = 0; i < nums.size(); i++) {

			// 找到要插入的位置
			int j = 0;
			while (j < sonums.size() && sonums[j] < nums[i]) { j++; };
			sonums.insert(sonums.begin() + j, nums[i]);
			result.insert(result.end(), j);
		}

		std::reverse(result.begin(), result.end());
		return result;


	}
	// 316.
	string removeDuplicateLetters(string s) {
		map<char, int > indexlast;
		string result = "";

		for (int i = 0; i < s.size(); i++) {
			if (indexlast.count(s[i])>0) {
				indexlast[s[i]] = i;
			}
			else {
				indexlast.insert(make_pair(s[i], i));
			}
		}

		for (int i = 0; i < s.size(); i++) {
			if (find(result.begin(), result.end(), s[i]) == result.end()) {
				while (result != "" && s[i] < result[result.size() - 1]
					&& i < indexlast[result[result.size() - 1]])
					result.pop_back();
				result.insert(result.end(), s[i]);
			}
		}

		return result;
	}

	//318.
	int maxProduct1(vector<string>& words) {
		vector<set<char>> words_set(words.size());
		vector<int> maxP{ 0 };

		for (int i = 0; i < words.size(); i++)
			words_set[i] = set<char>(words[i].begin(), words[i].end());

		for (int i = 0; i < words.size(); i++) {
			for (int j = i + 1; j < words.size(); j++) {
				set<char> temp;
				set_intersection(words_set[i].begin(), words_set[i].end(),
					words_set[j].begin(), words_set[j].end(), insert_iterator<set<char>>(temp, temp.begin()));
				if (temp.size() == 0) {
					maxP.push_back(words[i].size()*words[j].size());
				}
			}
		}
		return *max_element(maxP.begin(), maxP.end());
	}

	int maxProduct(vector<string>& words) {
		auto bit_num = [](char & ch) {return ch - 'a'; };
		map<long long, int> hashmap;

		for (int i = 0; i < words.size(); i++) {
			long long bitmask = 0;
			for (int j = 0; j < words[i].size(); j++)
				bitmask |= 1 << bit_num(words[i][j]);
			if (hashmap.count(bitmask) > 0)
				hashmap[bitmask] = max(hashmap[bitmask], (int)words[i].size());
			else
				hashmap.insert(make_pair(bitmask, (int)words[i].size()));
		}
		int max_pro = 0;

		for (auto iteri = hashmap.begin(); iteri != hashmap.end(); iteri++)
			for (auto iterj = hashmap.begin(); iterj != hashmap.end(); iterj++)
				if ((iteri->first & iterj->first) == 0)
					max_pro = max(max_pro, iteri->second * iterj->second);

		return max_pro;
	}
	// 322.
	int coinChange(vector<int>& coins, int amount) {
		vector<int> dp(amount + 1, amount + 1);
		int max = amount + 1;
		dp[0] = 0;
		for (int i = 1; i < amount + 1; i++)
			for (int j = 0; j < coins.size(); j++)
				if (coins[j] <= i)
					dp[i] = min(dp[i], dp[i - coins[j]] + 1);

		return dp[amount] > amount ? -1 : dp[amount];

	}
	//324
	void wiggleSort(vector<int>& nums) {
		sort(nums.begin(), nums.end());
		int len = nums.size();
		vector<int> temp1(nums.begin(),
			nums.begin() + (len - 1) / 2 + 1);
		vector<int> temp2(nums.begin() + (len - 1) / 2 + 1,
			nums.end());
		reverse_iterator<vector<int>::iterator> riter(temp1.end());
		for (int i = 0; i < len; i = i + 2) {
			nums[i] = *riter, riter++;
		}
		riter = reverse_iterator<vector<int>::iterator>(temp2.end());
		for (int i = 1; i < len; i = i + 2)
			nums[i] = *riter, riter++;
	}
	//326.
	bool isPowerOfThree(double n) {

		if (n <= 0) return false;
		while (n > 0) {
			if (n == 1) return true;
			n = n / 3.0;
			if (floor(n) != n) return false;
		}
		return false;
	}

	////328.
	//ListNode* oddEvenList(ListNode* head) {
	//	int flag = 1;
	//	ListNode* odd = new ListNode(0), *even = new ListNode(0);
	//	ListNode* oddptr = odd;
	//	ListNode* evenptr = even;
	//	while (head != NULL) {
	//		if (flag == 1) {
	//			oddptr->next = head;
	//			oddptr = oddptr->next;
	//			flag = 0;
	//		}
	//		else {
	//			evenptr->next = head;
	//			evenptr = evenptr->next;
	//			flag = 1;
	//		}
	//		head = head->next;
	//		oddptr->next = nullptr;
	//		evenptr->next = nullptr;
	//	}
	//	oddptr->next = even->next;
	//	delete even;
	//	head = odd->next;
	//	delete odd;
	//	return head;
	//}

	typedef vector<vector<int>> M;
	int dfs329(int x, int y, M& matrix, M& Pdeep) {
		int maxdeep = 0;
		M v{ vector<int>{x - 1,y},vector<int>{x + 1,y},vector<int>{x,y - 1},vector<int>{x,y + 1} };
		for (int k = 0; k < v.size(); k++) {
			int i = v[k][0];
			int j = v[k][1];
			if (i < 0 || i >= matrix.size() || j < 0 || j >= matrix[i].size())
				continue;
			if (matrix[i][j] > matrix[x][y]) {
				if (Pdeep[i][j] != 0)
					maxdeep = max(Pdeep[i][j], maxdeep);
				else {
					int temp = dfs329(i, j, matrix, Pdeep);
					Pdeep[i][j] = temp;
					maxdeep = max(temp, maxdeep);
				}
			}
		}
		return maxdeep + 1;
	}

	//329.Longest Increasing Path in a Matrix
	int longestIncreasingPath(M& matrix) {
		if (matrix.size() <= 0 || matrix[0].size() <= 0) return 0;
		int maxdeep = 0;
		M Pdeep(matrix.size(), vector<int>(matrix[0].size(), 0));
		for (int i = 0; i < matrix.size(); i++) {
			for (int j = 0; j < matrix[i].size(); j++) {
				if (Pdeep[i][j] != 0)
					continue;
				Pdeep[i][j] = dfs329(i, j, matrix, Pdeep);
				maxdeep = max(maxdeep, Pdeep[i][j]);
			}
		}
		return maxdeep;
	}

	//330.Patching Array
	int minPatches(vector<int>& nums, int n) {
		unsigned int m = 1, p = 0, i = 0, size = nums.size();
		while (m <= n) {
			if (i < size && nums[i] <= m)
				m += nums[i++];
			else
				m += m, p++;
		}
		return p;
	}

	typedef string T;
	//332.
	template<class T>
	vector<string> findItinerary(vector<vector<T>>& Egdes, T start = "JFK") {
		// 在有向多重图寻找一条不重复便利路径
		// 图是起点--->终点的边表示形式
		map<T, vector<T>> node_dicts; // 字典形式图
		map<tuple<T, T>, int> pathsum; // 每一条边的重复数
		int edges_size = Egdes.size();

		for (vector<string>& x : Egdes) {
			string &k = x[0];
			string &v = x[1];
			pathsum[make_tuple(k, v)] += 1;
			if (node_dicts.count(k) > 0)
				node_dicts[k].insert(node_dicts[k].end(), v);
			else
				node_dicts.insert({ k, vector<T>{v} });
		}

		for (auto& x : node_dicts) {
			vector<T> & list = x.second;
			sort(list.begin(), list.end());
		}

		auto result = findItineraryDFS<T>(start, vector<T>{}, map<tuple<T, T>, int>{},
			node_dicts, pathsum, edges_size);
		return result;

	}

	template<class T>
	vector<T> findItineraryDFS(T start, vector<T> path,
		map<tuple<T, T>, int>& pd, map<T, vector<T>>& node_dicts,
		map<tuple<T, T>, int>& pathsum, int size) {
		int sum = 0;
		for (auto& x : pd) {
			sum += x.second;
		}
		if (sum == size) {
			path.insert(path.end(), start);
			return path;
		}
		for (const auto& next : node_dicts[start]) {
			tuple<T, T> key{ start, next };
			if (pd.count(key) == 0)
				pd.insert({ key, 0 });

			if (pd[key] < pathsum[key]) {
				pd[key] += 1;
				path.insert(path.end(), start);
				auto result = findItineraryDFS<T>(next, path, pd, node_dicts,
					pathsum, size);
				if (result.size() > 0)
					return result;
				path.erase(path.end() - 1);
				pd[key] -= 1;
			}
		}
		return vector<T>{};
	}

	bool increasingTriplet(vector<int>& nums) {
		if (nums.size() <= 1)return false;
		int k = *max_element(nums.begin(), nums.end()) + 1;
		int n1 = k, n2 = k;
		for (int n : nums) {
			if (n < n1)
				n1 = n;
			else
				if (n < n2)
					n2 = n;
				else
					return true;

		}
		return false;

	}
	bool H335(int k, vector<int>& x) {
		vector<int> step;
		for (int i = 0; i < 6; i++) {
			if (k + i < x.size())
				step.push_back(x[k + i]);
			else
				step.push_back(0);
		}
		if (step[3] < step[1]) return false;
		if (step[2] <= step[0]) return true;
		if ((step[4] >= (step[2] - step[0])) &&
			(step[4] <= step[2]) && (step[5] >= step[3] - step[1]))
			return true;
		return false;
	}
	// 335.
	bool isSelfCrossing(vector<int>& x) {
		for (int i = 0; i < int(x.size() - 3); i++)
			if (H335(i, x))
				return true;
		return false;
	}

	//336.
	typedef pair<int, int> VT;
	VT robdfs(TreeNode* root) {
		if (root == nullptr)
			return{ 0, 0 };
		auto rl = robdfs(root->left);
		auto rr = robdfs(root->right);
		return{ max(root->val + rl.second + rr.second, rl.first + rr.first), rl.first + rr.first };
	}
	int rob(TreeNode* root) {
		VT result = robdfs(root);
		return max(result.first, result.second);
	}


	void WordsReverse() {
		string str;
		vector<string> sv;
		getline(cin, str);
		int i = 0, j = 0;
		while ((j = str.find(' ', i)) >= 0) {
			sv.insert(sv.begin(), str.substr(i, j - i));
			i = j + 1;
		}
		sv.insert(sv.begin(), str.substr(i, str.size() - i));
		// 这句话有问题
		//for_each(sv.begin(), sv.end(), [&](string& x) {reverse(begin(x), end(x)); });
		str = "";
		for (string& s : sv) {
			str = str + " " + s;
		}
		str = str.substr(1, str.size() - 1);
		cout << str << endl;
	}

	int fk(int start, int total, vector<int>& v) {
		if (total < 0)
			return 0;
		else {
			if (start == v.size() - 1)
			{
				if (v[start] > total) return 1;
				else return 2;
			}
			return fk(start + 1, total - v[start], v) + fk(start + 1, total, v);
		}
	}
};



ostream & operator<<(ostream& os, vector<int> v) {
	for (auto iter = v.cbegin(); iter != v.cend(); iter++)
		os << *iter << " ";
	return os;
}

// 341.	
/**
* // This is the interface that allows for creating nested lists.
* // You should not implement it, or speculate about its implementation
* class NestedInteger {
*   public:
*     // Return true if this NestedInteger holds a single integer, rather than a nested list.
*     bool isInteger() const;
*
*     // Return the single integer that this NestedInteger holds, if it holds a single integer
*     // The result is undefined if this NestedInteger holds a nested list
*     int getInteger() const;
*
*     // Return the nested list that this NestedInteger holds, if it holds a nested list
*     // The result is undefined if this NestedInteger holds a single integer
*     const vector<NestedInteger> &getList() const;
* };
*/

class NestedInteger {
public:
	// Return true if this NestedInteger holds a single integer, rather than a nested list.
	bool isInteger() const;

	// Return the single integer that this NestedInteger holds, if it holds a single integer
	// The result is undefined if this NestedInteger holds a nested list
	int getInteger() const;

	// Return the nested list that this NestedInteger holds, if it holds a nested list
	// The result is undefined if this NestedInteger holds a single integer
	const vector<NestedInteger> &getList() const;

};
class NestedIterator {
private:
	int label;
	vector<int> nv;
public:
	NestedIterator(vector<NestedInteger> &nestedList) {
		label = 0;
		for (int i = 0; i < nestedList.size(); i++) {
			if (nestedList[i].isInteger()) {
				//下一个数为整数
				nv.push_back(nestedList[i].getInteger());
			}
			else {//为嵌套列表
				vector<NestedInteger> temp = nestedList[i].getList();
				NestedIterator temp_back(temp);
				vector<int> nv_back = temp_back.getnv();
				copy(nv_back.begin(), nv_back.end(), back_inserter(nv));
			}
		}
	}

	vector<int> getnv() {
		return nv;
	}

	int next() {
		return nv[label++];
	}

	bool hasNext() {
		return label != nv.size();
	}
};
//344.
void reverseString(vector<char>& s) {
	int i = 0, j = s.size() - 1;
	while (i < j) {
		swap(s[i++], s[j--]);
	}
}
//345.
string reverseVowels1(string s) {
	//元音
	vector<char> sc;
	vector<char> vowels{ 'a','e','i','o','u' };
	vector<int> seq;

	copy(begin(s), end(s), back_inserter(sc));

	for (int i = 0; i < sc.size(); i++) {
		char temp = sc[i];
		if (temp < 93) temp += 32;
		auto lo = find(vowels.begin(), vowels.end(), temp);
		if (lo != vowels.end()) {
			seq.push_back(i);
		}
	}

	sort(seq.begin(), seq.end());
	int i = 0, j = seq.size() - 1;
	while (i < j) {
		swap(sc[i++], sc[j--]);
	}

	s = "";
	for_each(sc.rbegin(), sc.rend(), [&s](char &ch) {s += ch; });
	return s;
}

string reverseVowels(string s) {
	vector<char> vowels{ 'a','e','i','o','u' };
	int front = 0, rear = s.size() - 1;
	string newsl = "", newsr = "";
	while (front < rear) {
		int flo = -1;
		if (s[front] < 93)
			flo = find(vowels.begin(), vowels.end(), s[front] + 32) - vowels.begin();
		else
			flo = find(vowels.begin(), vowels.end(), s[front]) - vowels.begin();
		if (flo != vowels.size()) {
			// 左侧找到元音
			// 寻找右侧元音字母
			while (front < rear) {
				int rlo = -1;
				if (s[rear] < 93)
					rlo = find(vowels.begin(), vowels.end(), s[rear] + 32) - vowels.begin();
				else
					rlo = find(vowels.begin(), vowels.end(), s[rear]) - vowels.begin();
				if (rlo != vowels.size()) {
					// 右侧找到一个元音
					newsl += s[rear--]; newsr = s[front++] + newsr;
					break;
				}
				else {
					newsr = s[rear--] + newsr;
					if (rear == front) {
						newsl += s[front++];
					}
				}
			}
		}
		else {
			newsl += s[front++];
			if (front == rear) {
				newsr = s[rear--] + newsr;
			}
		}
	}
	return newsl + newsr;
}

vector<int> topKFrequent(vector<int>& nums, int k) {

	map<int, int> frequent;
	for (int i = nums.size() - 1; i >= 0; i--)
		if (frequent.count(nums[i]) == 0)
			frequent.insert({ nums[i],1 });
		else
			frequent[nums[i]]++;
	vector<pair<int, int>> vs(frequent.begin(), frequent.end());
	sort(vs.begin(), vs.end(), [](const pair<int, int>& v1, const pair<int, int>& v2) {
		return v1.second > v2.second;
	});
	vector<int> result;
	for (int i = 0; i < k; i++) {
		result.push_back(vs[i].first);
	}
	return result;
}
vector<int> intersection(vector<int>& nums1, vector<int>& nums2) {
	set<int> n1(nums1.begin(), nums1.end()), n2(nums2.begin(), nums2.end());
	vector<int> result;
	set_intersection(n1.begin(), n1.end(), n2.begin(), n2.end(), back_inserter(result));
	return result;
}
vector<int> intersect(vector<int>& nums1, vector<int>& nums2) {
	multiset<int> n1(nums1.begin(), nums1.end()), n2(nums2.begin(), nums2.end());
	vector<int> result;
	set_intersection(n1.begin(), n1.end(), n2.begin(), n2.end(), back_inserter(result));
	return result;
}

//354.
int maxEnvelopes(vector<vector<int>>& envelopes) {
	auto start_t = clock();
	typedef vector<int> SN;
	map<SN, vector<SN>> dict;
	map<SN, vector<SN>> in;
	map<SN, int> pathlen;
	vector<SN> start;
	for (int i = envelopes.size() - 1; i >= 0; i--)
	{
		if (dict.count(envelopes[i]) == 0) {
			dict.insert({ envelopes[i],vector<SN>() }),
				in.insert({ envelopes[i],vector<SN>() });
		}
		pathlen.insert({ envelopes[i], 1 });
		for (int j = i - 1; j >= 0; j--) {
			if (envelopes[i][0] - envelopes[j][0] > 0 && envelopes[i][1] - envelopes[j][1] > 0)
				dict[envelopes[j]].push_back(envelopes[i]),
				in[envelopes[i]].push_back(envelopes[j]);
			else
				if (envelopes[j][0] - envelopes[i][0] > 0 && envelopes[j][1] - envelopes[i][1] > 0)
					dict[envelopes[i]].push_back(envelopes[j]),
					in[envelopes[j]].push_back(envelopes[i]);
		}
	}
	for (auto iter = dict.begin(); iter != dict.end(); iter++)
		if (iter->second.size() == 0)
			start.push_back(iter->first);

	while (start.size() != 0) {
		vector<SN> temp;
		for (int i = start.size() - 1; i >= 0; i--) {
			for (auto iter = in[start[i]].begin(); iter != in[start[i]].end(); iter++) {
				dict[*iter].erase(find(dict[*iter].begin(), dict[*iter].end(), start[i]));
				pathlen[*iter] = max(1 + pathlen[start[i]], pathlen[*iter]);
				if (dict[*iter].size() == 0)
					temp.push_back(*iter);
			}
		}
		start = temp;
	}
	int result = 0;
	for (auto iter = pathlen.begin(); iter != pathlen.end(); iter++)
		result = max(result, iter->second);
	clock_t end_t = clock();
	cout << "time : " << end_t - start_t << endl;
	return result;
}
//355.
class Twitter {
public:
	/** Initialize your data structure here. */
	Twitter() {

	}

	/** Compose a new tweet. */
	void postTweet(int userId, int tweetId) {

	}

	/** Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent. */
	vector<int> getNewsFeed(int userId) {

	}

	/** Follower follows a followee. If the operation is invalid, it should be a no-op. */
	void follow(int followerId, int followeeId) {

	}

	/** Follower unfollows a followee. If the operation is invalid, it should be a no-op. */
	void unfollow(int followerId, int followeeId) {

	}
};

/**
* Your Twitter object will be instantiated and called as such:
* Twitter* obj = new Twitter();
* obj->postTweet(userId,tweetId);
* vector<int> param_2 = obj->getNewsFeed(userId);
* obj->follow(followerId,followeeId);
* obj->unfollow(followerId,followeeId);
*/

//357.
int countNumbersWithUniqueDigits(int n) {
	int f1 = 10;
	int f2 = 0;
	while (n-- != 1) {
		f2 = 9;
		for (int i = 0; i < n; i++) {
			f2 *= (9 - i);
		}
		f1 += f2;
	}
	return f1;
}



//367.
bool isPerfectSquare(int num) {
	double sn = sqrt(num);
	return floor(sn) == sn;
}

//371.
int getSum(int a, int b) {
	return a + b;
}

int guess(int num) {
	int key = 6000121242124211;
	if (num == key)
		return 0;
	if (num > key)
		return 1;
	else
		return -1;
}

//374.
int guessNumber(int n) {
	int low = 1, high = 2;
	while ((guess(high)) < 1) {
		low = high, high *= 2;
	}

	int mid = 0, g = 0;
	while (low < high) {
		cout << low << " " << high << " " << mid << endl;
		mid = (low + high) / 2;
		g = guess(mid);
		if (g == 0) return mid;
		if (g == 1) high = mid;
		if (g == -1) low = mid;
	}
	return -1;
}
//368.
vector<int> largestDivisibleSubset(vector<int>& nums) {
	if (nums.size() == 0) return vector<int>();
	sort(nums.begin(), nums.end());

	//FList 存储当前位置的最大结果
	vector<vector<int>> FList(nums.size(), vector<int>());
	int maxcount = 0;
	for (int i = nums.size() - 1; i >= 0; i--) {
		FList[i].push_back(nums[i]);
		int temp = i;
		for (int j = i + 1; j < nums.size(); j++) {
			temp = nums[j] % nums[i] == 0 && FList[j].size() >= FList[temp].size()
				? j : temp;
		}
		if (i != temp)
			for_each(FList[temp].begin(), FList[temp].end(),
				[&](int x) {FList[i].push_back(x); });
		maxcount = FList[i].size() > FList[maxcount].size() ? i : maxcount;
	}
	return FList[maxcount];
}


int quick_pow(int a, int b, int m) { // 快速幂
	int ans = 1;
	while (b > 0) {
		if (b & 1)
			ans = (ans*a) % m;
		a = (a*a) % m;
		b >>= 1;
	}
	return ans%m;
}

int euler(int n) {//欧拉公式
	int ret = n;
	for (int i = 2; i*i < n; i++) {
		if (n%i == 0)// i为n的质因素
		{
			ret = ret / n*(n - 1);
			while (n%i == 0)
				n /= i;
		}
	}
	if (n > 1) {// n 本来就是指数 f(n) = n-1
		ret = ret / n*(n - 1);
	}
	return ret;
}

//372.
int superPow(int a, vector<int>& b) {
	//a * b
	//return a *b mode 1337

	int c = 1337;
	int exp = 0;
	int phi = euler(c);

	for (int i = 0; i < b.size(); i++)
		exp = (exp * 10 + b[i]) % phi;

	return quick_pow(a, exp, c);
}
class comp {
	//仅适用于373.
public:
	bool operator()(vector<int> x, vector<int> y) {
		return (x[0] + x[1]) <= (y[0] + y[1]) ? true : false;
	}
};
//373.
vector<vector<int>> kSmallestPairs(vector<int>& nums1, vector<int>& nums2, int k) {
	typedef vector<int> vi;
	k = min(int(nums1.size()*nums2.size()), k);
	priority_queue<vi, vector<vi>, comp> pri;
	for (int x = 0; x < nums1.size(); x++) {
		for (int y = 0; y < nums2.size(); y++)
			pri.push(vi{ nums1[x], nums2[y] });
	}
	vector<vi> result;
	for (int i = 0; i < k; i++)
	{
		result.push_back(pri.top());
		pri.pop();
	}
	return result;

}
//375.
int getMoneyAmount2(int n) {
	//pass
	return 0;
}
//376.
int wiggleMaxLength(vector<int>& nums) {
	if (nums.size() <= 1) return nums.size();
	while (nums[1] == nums[0] && nums.size()>1)
		nums.erase(nums.begin());
	if (nums.size() <= 2) return nums.size();
	int pre = nums[1] > nums[0] ? 1 : -1, next = 1;

	for (auto iter = nums.begin() + 2; iter != nums.end(); iter++) {
		if (*(iter) == *(iter - 1)) {
			nums.erase(iter), iter--;
			continue;
		}
		next = (*(iter) > *(iter - 1)) ? 1 : -1;
		if (pre == next)
			nums.erase(iter - 1), iter--;
		pre = next;
	}

	return nums.size();
}
//377. 
int combinationSum4(vector<int>& nums, int target) {
	//暴力动态规划，我觉得很慢
	vector<int> dp(target + 1, 0);
	for (int i = 1; i < target + 1; i++) {
		for (size_t j = 0; j < nums.size(); j++) {
			if (i == nums[j])
				dp[i] += 1;
			if (i > nums[j])
				dp[i] += dp[i - nums[j]];
		}
	}
	return dp[target];
}

//387.
int firstUniqChar(string s) {
	vector<int> d(26, 0);
	for (char c : s)
		d[c - 'a']++;
	string news;
	for (int i = 0; i < d.size(); i++)
		if (d[i] == 1) {
			char c = i + 'a';
			news += c;
		}

	int index = s.find_first_of(news);
	return index;
}
//389.
char findTheDifference(string s, string t) {
	sort(s.begin(), s.end());
	sort(t.begin(), t.end());
	for (int i = 0; i < s.size(); i++)
		if (s[i] != t[i])
			return s[i];
	return t.back();
}

//404.
int sumOfLeftLeaves(TreeNode* root) {
	int count = 0;
	vector<TreeNode* > stack;

	while (!stack.empty() || root) {
		while (root->left) stack.push_back(root), root = root->left;
		root = stack.back();
		stack.pop_back();
		if (root->right == nullptr)
			count += root->val;

		stack.push_back(root->right);
	}

}
//378.
int kthSmallest(vector<vector<int>>& matrix, int k) {
	vector<int>nums;
	for (auto &ma : matrix) {
		for (int x : ma)
			nums.push_back(x);
	}
	nth_element(nums.begin(), nums.begin() + k - 1, nums.end());
	return nums[k - 1];
}

//380.

class RandomizedSet {
public:
	/** Initialize your data structure here. */
	RandomizedSet() {
	}

	/** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
	bool insert(int val) {
		if (result.count(val) == 0) {
			result.insert({ val, val });
			return true;
		}
		else
			return false;
	}

	/** Removes a value from the set. Returns true if the set contained the specified element. */
	bool remove(int val) {
		if (result.count(val) != 0) {
			result.erase(val);
			return true;
		}
		else
			return false;
	}

	/** Get a random element from the set. */
	int getRandom() {
		vector<pair<int, int>> list(result.begin(), result.end());
		int random = rand() % (result.size());
		return list[random].first;
	}

private:
	map<int, int> result;
};

//382. - 1
class Solution3821 {
public:
	/** @param head The linked list's head.
	Note that the head is guaranteed to be not null, so it contains at least one node. */
	Solution3821(ListNode* head) :head(head) {
		List_size = -1;
	}

	/** Returns a random node's value. */
	int getRandom() {
		if (List_size < 0) {
			auto p = head;
			while (p) {
				List_size++;
				p = p->next;
			}
		}
		int step = rand() % List_size;
		auto p = head;
		while (step--) p = p->next;
		return p->val;
	}
private:
	ListNode *head;
	int List_size;
};

//382.
class Solution382 {
public:
	/** @param head The linked list's head.
	Note that the head is guaranteed to be not null, so it contains at least one node. */
	Solution382(ListNode* head) {
		ListNode* p = head;
		while (p) {
			List.push_back(p->val);
			p = p->next;
		}
	}

	/** Returns a random node's value. */
	int getRandom() {
		int index = rand() % List.size();
		return List[index];
	}
private:
	vector<int> List;
};

//384
class Solution384 {
public:
	Solution384(vector<int>& nums) :nums(nums), stactic_nums(nums) {
	}

	/** Resets the array to its original configuration and return it. */
	vector<int> reset() {
		return stactic_nums;
	}

	/** Returns a random shuffling of the array. */
	vector<int> shuffle() {
		random_shuffle(nums.begin(), nums.end());
		return nums;
	}
private:
	vector<int> nums, stactic_nums;
};

//386. Lexicographical Numbers
typedef pair<string, int> P;
bool Com381(const int& a, const int& b) {
	return to_string(a) < to_string(b);
}

vector<int> lexicalOrder(int n) {
	vector<int> list(n - 1);
	iota(list.begin(), list.end(), 1);
	sort(list.begin(), list.end(), Com381);
	return list;
}

//391.

//填充
void FillArea(vector<vector<int>>& area, const vector<int>& rect) {
	for (int j = rect[1]; j <= rect[3]; j++)
		for (int i = rect[0]; i <= rect[2]; i++)
			area[i][j] = 1;
}

//暴力填充
bool isRectangleCover(vector<vector<int>>& rectangles) {
	typedef pair<int, int> point;
	set<point> hp;

	int area = 0;
	for (int i = 0; i < rectangles.size(); i++) {
		vector<point> ps{ point{ rectangles[i][0],rectangles[i][1] },
			point{ rectangles[i][0],rectangles[i][3] },
			point{ rectangles[i][2],rectangles[i][1] },
			point{ rectangles[i][2],rectangles[i][3] } };
		area += (rectangles[i][2] - rectangles[i][0])*(rectangles[i][3] - rectangles[i][1]);
		for (int i = ps.size() - 1; i >= 0; i--) {
			auto iter = hp.find(ps[i]);
			if (iter == hp.end())
				hp.insert(ps[i]);
			else
				hp.erase(iter);
		}
	}

	if (hp.size() != 4)
		return false;
	vector<point> hr(hp.begin(), hp.end());
	sort(hr.begin(), hr.end(), [](const point& a, const point& b) {
		if (a.first < b.first) return true;
		else if (a.first == b.first)
			return a.second < b.second;
		else return false;
	});
	return area == (hr[0].second - hr[3].second)*(hr[0].first - hr[3].first);
}

string decodeString(string s) {
	/*
		解析字符串：一段字符(比如：3[adsfa]或者直接是非重复字符串段)
		一段字符结束标志分别为 数字 或者 ] ，使用end交替标记，stack用来标记括号的对数，
		stack=0则表示发现一个完整的字符段
	*/
	int start = 0;
	char end = 'n';
	int stack = 0;
	string result = "";
	for (int i = 0; i < s.size(); i++) {
		if (end == 'n') {
			if (s[i] >= '0' && s[i] <= '9') {
				end = '[';
				result += s.substr(start, i - start);
				start = i;
			}
		}
		else {
			if (s[i] == '[')
				stack++;
			else if (s[i] == ']') {
				if (--stack == 0) {
					end = 'n';
					int ls = s.find('[', start); //查找新段的第一个'[
					string temp = decodeString(s.substr(ls + 1, i - ls - 1)); //递归解析中括号内字符串
					int times = stoi(s.substr(start, ls - start));
					for (int j = times; j > 0; j--)
						result += temp;
					start = i + 1;
				}
			}
		}
	}
	result += s.substr(start, s.size() - start);
	return result;
}

//395.
int longestSubstring(string s, int k) {
	map<char, int> AlphabetIndex; // 记录字符出现了几次
	map<char, vector<int>> Alphabetlocal; // 记录字符出现的位置
	int result = 0; // 结果
	set<char> lsA(s.begin(), s.end()); // 字符串s的集合，用于记录次数少于k次的字符
	for (int i = s.size() - 1; i >= 0; i--) {
		if (AlphabetIndex.count(s[i]) == 0)
			AlphabetIndex.insert({ s[i],1 });
		else
			AlphabetIndex[s[i]]++;
		if (Alphabetlocal.count(s[i]) == 0)
			Alphabetlocal.insert({ s[i], {i} });
		else
			Alphabetlocal[s[i]].push_back(i);
	}
	vector<int> index{ -1 };
	for (map<char, int>::iterator iter = AlphabetIndex.begin();
	iter != AlphabetIndex.end(); iter++) {
		if (iter->second < k)
			for (int j = Alphabetlocal[iter->first].size() - 1; j >= 0; j--)
				index.push_back(Alphabetlocal[iter->first][j]);
	}
	index.push_back(s.size());
	sort(index.begin(), index.end());
	if (index.size() == 2)
		return s.size();
	for (int i = 0; i < index.size() - 1; i++)
		result = max(result,
			longestSubstring(
				s.substr(index[i] + 1, index[i + 1] - index[i] - 1), k));
	return result;
}

//397.
int integerReplacement(long long  n) {
	if (n == 1) return 0;
	if (n == 2) return 1;
	if (n % 2 == 1)
		return min(integerReplacement(n + 1), integerReplacement(n - 1)) + 1;
	else
		return integerReplacement(n / 2) + 1;
}
//398.
class Solution398 {
private:
	vector<int> nums;
	map<int, vector<int>> dict;
public:
	Solution398(vector<int>& nums) :nums(nums) {
		for (int i = nums.size() - 1; i >= 0; i--) {
			if (dict.count(nums[i]) == 0)
				dict.insert({ nums[i], vector<int>{i} });
			else
				dict[nums[i]].push_back(i);
		}
	}

	int pickMoreTimes(int target) {
		const auto& v = dict[target];
		int randomIndex = rand() % v.size();
		return v[randomIndex];
	}

	int pickOneTime(int target) {
		int d = 1;
		int res;
		for (int i = nums.size() - 1; i >= 0; i--) {
			if (nums[i] == target) {
				if (rand() % d == 0)
					res = i;
				d++;
			}
		}
		return res;
	}
};

//399.
class Fraction { // 简易 分数结构
private:
	double numerator; //分子
	double denominator; // 分母
public:
	Fraction() :numerator(1), denominator(1) {}
	Fraction(double x, double y = 1) :numerator(x), denominator(y) {}
	double Ftod() {
		return numerator / denominator;
	}
	friend Fraction operator* (const Fraction& lf, const Fraction& rf);
};
Fraction operator* (const Fraction& lf, const Fraction& rf) {
	return Fraction(lf.numerator*rf.numerator, lf.denominator*rf.denominator);
}

double FindcalcBFS(map<string, vector<string>>& edges, map <pair<string, string>, Fraction>& weight, string& x, string& y) {
	if (weight.count({ x,y }) == 1) {
		return  weight[{x, y}].Ftod();
	}
	vector<string> stack{ x };
	set<string> path{x};
	bool flag = true;
	while (!stack.empty() && flag) {
		string temp = stack.back();
		stack.pop_back();
		for (int i = 0; i < edges[temp].size(); i++) {
			if (path.count(edges[temp][i]) == 0) {
				if (weight.count({ x, edges[temp][i] }) == 0)
					weight.insert({ { x, edges[temp][i] }, weight[{x, temp}] * weight[{temp,  edges[temp][i]}] });
				stack.push_back(edges[temp][i]);
				path.insert(edges[temp][i]);
			}
			if (edges[temp][i] == y) {
				return weight[{x, y}].Ftod();
			}
		}
	}
	return -1.0;
}

vector<double> calcEquation(vector<vector<string>>& equations, vector<double>& values, vector<vector<string>>& queries) {
	map <pair<string, string>, Fraction> weight;
	map<string, vector<string>> edges;
	for (int i = equations.size() - 1; i >= 0; i--) {
		pair<string, string > key(equations[i][0], equations[i][1]);
		if (weight.count(key) == 0) {
			weight.insert({ key, Fraction(values[i]) });
			weight.insert({ { equations[i][1], equations[i][0] }, Fraction(1, values[i]) });
		}
		if (edges.count(equations[i][0]) == 0) {
			edges.insert({ equations[i][0] ,{ equations[i][1] } });
		}
		else{
			edges[equations[i][0]].push_back(equations[i][1]);
			}
		if (edges.count(equations[i][1]) == 0) {
			edges.insert({ equations[i][1] ,{ equations[i][0] } });
		}
		else {
			edges[equations[i][1]].push_back(equations[i][0]);
		}
	}
	vector<double> results;
	for (vector<vector<string>>::iterator iter = queries.begin(); iter != queries.end(); iter++) {
		
		if (edges.count((*iter)[0]) == 0 || edges.count((*iter)[1]) == 0)
			results.push_back(-1.0); 
		else if((*iter)[0]==(*iter)[1])
			results.push_back(1.0);
		else
			results.push_back(FindcalcBFS(edges, weight, (*iter)[0], (*iter)[1]));
	}
	return results;
}

int CaluSum(bitset<8> b) {
	//计算数字二进制中1的个数
	int result=0;
	for (int i = 0; i < 8; i++)
		result += b[i];
	return result;
}

void CalueDict(int num, double MAX, map<int, vector<int>>& dict) {
	// 计算数字二进制中1的个数字典
	dict.insert({ 0, {0} });
	for (int i = 1; i < min(MAX, pow(2, num)); i++) {
		auto b = bitset<8>(i);
		int s = CaluSum(b);
		if (dict.count(s) == 0) {
			dict.insert({ s, {i} });
		}
		else {
			dict[s].push_back(i);
		}
	}

}

string getresult(int lt, int rt) {
	// 格式化字符时间
	string result = to_string(lt)+":";
	if (rt <= 9) {
		result += "0" + to_string(rt);
	}
	else {
		result += to_string(rt);
	}
	return result;
}

vector<string> CaluWatch(int num, map<int, vector<int>>& dict) {
	// 计算表盘时间
	vector<string> result;
	for (int i = 0; i < min(num + 1, 4); i++) {
		const auto& left = dict[i];
		for (auto lt = left.begin(); lt != left.end(); lt++) {
			if (*lt >= 12) continue;
			const auto& right = dict[num - i];
			for (auto rt = right.begin(); rt != right.end(); rt++) {
				result.push_back(getresult(*lt, *rt));
			}
		}
	}
	return result;
}

//401
vector<string> readBinaryWatch(int num) {
	map<int, vector<int>> dict;
	CalueDict(6, 60, dict); 
	return CaluWatch(num, dict);
}


static int x = []() {
	ios::sync_with_stdio(false);
	cin.tie(NULL);
	return 0;
}();


int main() {
	auto result = readBinaryWatch(0);
	copy(result.begin(), result.end(), ostream_iterator<string>(cout, " "));
}


