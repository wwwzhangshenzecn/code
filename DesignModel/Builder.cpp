/*
	��һ�����Ӷ���Ĺ��������ʾ����룬ʹ��ͨ�õĹ������̿��Դ�����ͬ�ı�ʾ��
*/
class House {
public:
	void Init() {
		this->B1();
		this->B2();
		this->B3();
		this->B4();
	}

protected:
	virtual B1() = 0;
	virtual B2() = 0;
	virtual B3() = 0;
	virtual B4() = 0;
};

class StoneHouse :public House {
protected:
	virtual B1() {};
	virtual B2() {};
	virtual B3() {};
	virtual B4() {};
};