/*
使用原型实例指定创建对象的种类，然后通过拷贝这些原型创建新的对象
*/

// 通过直接克隆自己，来达到于工厂模式相同的目的
class Base {
	..;
	virtual Base* clone() = 0;

};

class PC :public Base {

public:
	PC(PC* a){
		if (this != a)
			this->...
			this->...
		}
	} //复制构造

	virtual Base* clone() {
		return new PC(*this);
	}

};
