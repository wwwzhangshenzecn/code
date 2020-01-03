//将一个类的接口转换成客户夕阳的另外一个接口，
//Adapter模式使得原本由于接口不够兼容二不能一起工作的那些类可以
//一起工作

//目标接口（新接口）
class ITarget {
public:
	virtual void process() = 0;
};

//遗留接口（老接口）
class IAdaptee {
public:
	virtual void foo(int data = 0) = 0;
	virtual int bar() = 0;
};

class Adapter :public ITarget{
protected:
	IAdaptee* pAdaptee;
public:
	virtual void process() {
		//...
		int data=pAdaptee->bar();
		//...
		pAdaptee->foo(data);
		//...
	}
		


};


















