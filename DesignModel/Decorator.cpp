/*
��̬����ϣ��ظ�һ����������һЩ�����ְ�����ӹ��ܵ����ݡ�Decorator ģʽ������
�����Ϊ���������ظ����ࣩ
*/

class Base {
public:
	Base() {};
	virtual ~Base() {};
	virtual void f() = 0 {};
};

class Decorator {
protected:
	Base * b;
public:
	Decorator(Base *b) :b(b) {}
	~Decorator() {};
}

//��Base��������й�����չ
class XBase :public Decorator {
public:
	XBase(Base * b):Decorator(b) {}
	void f() {
		b->f();
	}
};
