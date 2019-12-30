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
