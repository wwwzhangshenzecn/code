class Base {
public:
	Base() {};
	virtual ~Base() {};
	virtual void f() = 0 {};
};

//直接利用组合对Base进行横向扩展

class ExtendBase :public Base {
public:
	//...
private:
	Base * b;

};