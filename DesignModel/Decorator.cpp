/*
动态（组合）地给一个对象增加一些额外的职责，增加功能的内容。Decorator 模式比生成
子类更为灵活。（消除重复子类）
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

//对Base的子类进行功能扩展
class XBase :public Decorator {
public:
	XBase(Base * b):Decorator(b) {}
	void f() {
		b->f();
	}
};
