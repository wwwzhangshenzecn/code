//��һ����Ľӿ�ת���ɿͻ�Ϧ��������һ���ӿڣ�
//Adapterģʽʹ��ԭ�����ڽӿڲ������ݶ�����һ��������Щ�����
//һ����

//Ŀ��ӿڣ��½ӿڣ�
class ITarget {
public:
	virtual void process() = 0;
};

//�����ӿڣ��Ͻӿڣ�
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


















