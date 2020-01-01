/*
定义一个操作中算法的骨架而将一些步骤延迟到子类中，Tempplate Method使子类
可以不改变一个算法的结构即可重定义该算法的某些特定步骤。
*/
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

