// ͨ��ֱ�ӿ�¡�Լ������ﵽ�ڹ���ģʽ��ͬ��Ŀ��
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
	} //���ƹ���

	virtual Base* clone() {
		return new PC(*this);
	}

};
