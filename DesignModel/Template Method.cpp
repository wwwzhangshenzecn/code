class Base {
public:
	Base() {};
	virtual ~Base() {};
	void f1() {};
	virtual void f2() = 0 {};
	void f3() {};
	virtual run() {
		f1();
		f2();
		f3();
	}
};

class EBase :public Base {
public:
	virtual void f2() {};
};

//  流程固定，但是其中一部分为变化，在运行过程中利用多态进行动态绑定
EBase().run()

