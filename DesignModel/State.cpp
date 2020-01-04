/*
����һ�����������ڲ�״̬�ı��Ǹı�������Ϊ��
���������ƺ��޸��������ࡣ
����ʱ�޸Ķ�������͡�
*/

//enum NetworkSate{Network_open, Network_cloes, Network_connection};

class NetworkState {
protected:
	NetworkState* next;
public:
	virtual void OP1() = 0;
	virtual void OP2() = 0;
	virtual void OP3() = 0;
};

class OpenState :public NetworkState {
private:
	static NetworkState* Instance;
public:
	OpenState(){}
	virtual void OP1() {
		//...
		next = CloseState::getInstance()
	}
	virtual void OP2() = 0;
	virtual void OP3() = 0;
	static NetworkState* getInstance() {
		if (Instance == nullptr) {
			Instance = new OpenState();
		}
		return Instance;
	}
};

class CloseState :public NetworkState {
private:
	static NetworkState* Instance;
public:
	virtual void OP1() {
		//...
		next =

	}
	virtual void OP2() = 0;
	virtual void OP3() = 0;
	static NetworkState* getInstance() {
		if (Instance == nullptr) {
			Instance = new CloseState();
		}
		return Instance;
	}
};

class ConnectState :public NetworkState {
private:
	static NetworkState* Instance;
public:
	virtual void OP1() {
		//...
		next =

	}
	virtual void OP2() = 0;
	virtual void OP3() = 0;
	static NetworkState* getInstance() {
		if (Instance == nullptr) {
			Instance = new ConnectState();
		}
		return Instance;
	}
};


class NetworkProcess:public NetworkState{
private:
	NetworkState* state;
public:
	NetworkProcess(NetworkState* state) :state(state) {}

	virtual void OP1() {
		state->OP1();
		state = state->next; //״̬ת��
		state->OP2();//ת��״̬���op2����
	}
	
};