/*
定义一系列算法，把他们一个个封装起来，并使它们可相互替换，该模式使得算法可独立于使用
他的客户程序。

*/
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