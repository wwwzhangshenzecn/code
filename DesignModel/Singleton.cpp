class singleton {
private:
	singleton();
	singleton(singleton s) {}
public:
	static singleton* getInstance();
	static singleton* instance;

};

static singleton* singleton::instance=nullptr;


//�̷߳ǰ�ȫ
singleton* singleton::getInstance() {
	if (instance == nullptr) {
		instance = new singleton();
	}
	return instance;
}

//�̰߳�ȫ�汾���������Ĵ��۹���
singleton* singleton::getInstance() {
	Lock lock;
	if (lock is yes):
		return instance;
	instance = new singleton();
	return instance;
}

//˫����������������ڴ��дreorder����ȫ
singleton* singleton::getInstance() {
	if (instance == nullptr) {
		Lock lock;
		if(instance == nullptr)
			instance = new singleton();
	}
	return instance;
}

#include<atomic>
#include<mutex>
std::atomic<singleton*> singleton::instance;
std::mutex singleton::mutex;


singleton* singleton::getInstance() {
	singleton* temp = instance.load(std::memory_order_relaxed);
	std::_Atomic_thread_fence(std::memory_order_acquire);
	if (temp == nullptr) {
		std::lock_guard<std::mutex> lock(mutex);
		temp = instance.load(std::memory_order_relaxed);
		if (temp == nullptr) {
			temp = new singleton;
			std::_Atomic_thread_fence(std::memory_order_release);
			instance.store(temp, std::memory_order_relaxed);
		}
	}
	return temp;
}
