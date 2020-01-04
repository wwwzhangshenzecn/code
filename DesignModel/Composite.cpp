#include<list>
/*
将对像组合成树形结构以表示“部分-整体”的层次结构。Composite使得用户对单个
对象和组合对象的使用具有一致性
*/
class Component {
public:
	virtual process() = 0;
};

class composite :public Component {
private:
	list<Component*> com;
	//....
public:
	virtual process() {
		// 处理当前节点
		for each(Component& c in com) {
			c->precess();
		}
	}

	void addElement(Component* c) {
		com.push_back(c);
	}

	void delEement(Component* c) {
		com.delete(c);
	}
};

class leaf :public Component {
private:
	//...
public:
	virtual process() {
		//...
	}
};

