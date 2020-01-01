// 通过直接克隆自己，来达到于工厂模式相同的目的
class Base {
	..;
	virtual Base* clone() = 0;

};

class PC :public Base {

public:
	PC(PC* a){
		if (this == a)
			return *this;
		else {
			this->...
			this->...
		}
	} //复制构造

	virtual Base* clone() {
		return new PC(*this);
	}

};
