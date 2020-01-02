class singleton {
private:
	singleton();
	singleton(singleton s) {}
public:
	static singleton* getInstance();
	static singleton* instance;

};

static singleton* singleton::instance=nullptr;


//线程非安全
singleton* singleton::getInstance() {
	if (instance == nullptr) {
		instance = new singleton();
	}
	return instance;
}

//线程安全版本，但是锁的代价过高
singleton* singleton::getInstance() {
	Lock lock;
	if (lock is yes):
		return instance;
	instance = new singleton();
	return instance;
}

//双检查锁，但是由于内存读写reorder不安全
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
