#include<list>
/*
��������ϳ����νṹ�Ա�ʾ������-���塱�Ĳ�νṹ��Compositeʹ���û��Ե���
�������϶����ʹ�þ���һ����
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
		// ����ǰ�ڵ�
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

