/*
允许一个对象在其内部状态改变是改变它的行为。
对象看起来似乎修改了它的类。
运行时修改对象的类型。
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
		state = state->next; //状态转换
		state->OP2();//转换状态后的op2操作
	}
	
};