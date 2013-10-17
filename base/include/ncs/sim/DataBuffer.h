#pragma once
#include <atomic>
#include <condition_variable>
#include <iostream>
#include <list>
#include <mutex>
#include <queue>
#include <stack>
#include <map>

#include <ncs/sim/Memory.h>

namespace ncs
{

namespace sim
{

//Forward Declarations
class DataBuffer;
class Publisher;
template<class PublicationType, class PublisherType> class Subscription;
template<typename T> class SpecificPublisher;

class Mailbox {
public:
	Mailbox();
	std::condition_variable arrival;
	std::mutex mutex;
	bool failed;
  template<typename... Pointers> bool wait(Pointers... pointers);
private:
  template<typename T, typename... OtherArgs>
  bool any_null_(T t, OtherArgs... o);
  template<typename T> bool any_null_(T** t);
  template<typename T> bool any_null_(std::vector<T*>* pointers);
};

/**
	An object that allows the owner to receive data buffers from Publishers.

	@param PublicationType What the Subscription receives
	@param PublisherType Where the Subscription receives PublicationTypes from
*/
template<class PublicationType, 
         class PublisherType = SpecificPublisher<PublicationType>>
class Subscription {
public:
	/**
		Constructor.

		@param publisher The publisher to subscribe to
	*/
	Subscription(PublisherType* publisher);

	/**
		Retrieves a data buffer that is ready for consumption. Waits for one
		to become available if none are currently.

		@return A publication once it is available
	*/
	PublicationType* pull();

	void pull(PublicationType** slot, Mailbox* mailbox);

	void cancel();

	/**
		Makes the buffer visible to the subscriber.

		@param data The publication to be read
	*/
	void push(PublicationType* data);

    /**
        Notifies the subscription that the publisher no longer exists.
    */
    void invalidate();

	/**
		Destructor: Unsubscribes from the source as well
	*/
	~Subscription();
private:
	///The mutex for the unique lock
	std::mutex mutex_;
	
	///Signals threads waiting on an available buffer or state change
	std::condition_variable state_changed_;
	
	///List of available buffers
	std::queue<PublicationType*> deliveries_;

	///The source of publications
	PublisherType* publisher_;

	Mailbox* mailbox_;

	PublicationType** mailslot_;
};

/**
	Base publishing interface. A publisher publishes information to a
	set of subscribers.
*/
class Publisher {
public:
	///Constructor
	Publisher();

	///Base subscription type.
	typedef class ncs::sim::Subscription<DataBuffer, Publisher> Subscription;

	/**
		Returns a subscription to the publisher's publications. Using this
		method yields publications that can only be accessed as a DataBuffer.

		@return A generic Subscription that genereates DataBuffers.
	*/
	Subscription* subscribe();

	/**
		Sends the DataBuffer to all subscribers. Returns the number of
		subscribers it was sent to.

		@param db The DataBuffer to publish
		@return The number of subscribers it was published to
	*/
	unsigned int publish(DataBuffer* db);

	/**
		Removes a subscription so that it can be safely deleted.

		@param sub The Subscription to remove
	*/
	void unsubscribe(Subscription* sub);

    /**
        Virtual destructor. Informs all subs that this Publisher is no more.
    */
  virtual ~Publisher();
	bool clearSubscriptions();

  bool setDevice(class DeviceBase* device);
  class DeviceBase* getDevice() const;
private:
  class DeviceBase* device_;
	///Lock around the subscriber list.
	std::mutex mutex_;
	
	///List of all subscribers
	std::list<Subscription*> subscriptions_;
};

/**
	More clearly defines the type of Publisher, allowing those objects that
	know the exact type of a Publisher to extract a more exact type of
	DataBuffer.
*/
template<typename T>
class SpecificPublisher : public Publisher {
public:
	///The more specific type of subscription
	typedef class ncs::sim::Subscription<T> Subscription;

	///Default Constructor
	SpecificPublisher();

	/**
		Returns a subscription to the more specific type of DataBuffer.

		@return A subscription that generates Ts
	*/
	Subscription* subscribe();

	/**
		Publishes the specific type of DataBuffer to all subscribers. Also
		publishes the generic DataBuffer to all generic subscribers.

		@param data The publication to push to subscribers
	*/
	unsigned int publish(T* data);

	/**
		Removes a subscriber from the list of subscriptions. All subsequent
		publications will not be delivered to it.

		@param sub The subscriber to unsubscribe
	*/
	void unsubscribe(Subscription* sub);

  /**
    Virtual destructor. Informs all specific subscribers that this
    publisher is no more.
   */
  virtual ~SpecificPublisher();

	/**
		Registers a DataBuffer for use with the publisher. Publishers should
		always request a blank, fill it with data, and then publish it.

		@param blank A fully allocated publication ready to be written to
	*/
	void addBlank(T* blank);

	/**
		Retrieves a blank that is not being consumed by any subscribers and
		not being used by the publisher already.

		@return A fully allocated publication ready to be written to
	*/
	T* getBlank();

private:
	///The mutex for the lock
	std::mutex mutex_;

	///List of specific subscribers
	std::list<Subscription*> subscriptions_;
	
	///Set of available data buffers for writing
	std::stack<T*> blanks_;
	
	///Signals waiting threads that a blank is available
	std::condition_variable blank_available_;

  ///The number of blanks this object manages
  unsigned int num_blanks_;
};

/**
	Base class for objects that can expose data for other objects to access.
*/
class DataBuffer {
public:
	/**
		Encapsulates a pointer to some abitrarily located piece of memory.
	*/
	class Pin
	{
  public:
		/**
			Default constructor for basically a null pin.
		*/
		Pin();

		/**
			Constructor

			@param d A pointer to the data to expose
			@param m The type of memory the data uses
		*/
		Pin(const void* data, DeviceType::Type memory_type);

    const void* getData() const;

    DeviceType::Type getMemoryType() const;
  private:
		///The pointer to the actual data
		const void* data_;

		///What memory system the data resides in
    DeviceType::Type memory_type_;
	};

	/**
		Returns the pin associated with a given key.

		@param key The name associated with the Pin
		@return The associated Pin if found; a null pin otherwise
	*/
	const Pin& getPin(const std::string& name) const;

	/**
		Acknowledge that the subscriber is done consuming data from this
		buffer.
	*/
	void release();

  bool setPrereleaseFunction(std::function<void()> prerelease_function);

  std::function<void()> getPrereleaseFunction() const;

	/**
		The function to call when release is called
	*/
	std::function<void()> release_function;

	///A count of all subscribers currently still consuming this buffer.
	std::atomic<unsigned int> subscription_count;

	///The number of threads that still need to update this buffer
	std::atomic<unsigned int> updates_needed;

	bool update();

  std::mutex* getWriteLock();

	///The simulation step this buffer was published from
	unsigned int simulation_step;
protected:
	/**
		Binds a pointer to a specific key.

		@param key The key to bind the data to
		@param data The data to bind
		@param memoryType The type of memory data exists on
		@return true if successful; false otherwise
	*/
	template<typename T>
	bool setPin_(const std::string& key,
               const T* data,
               DeviceType::Type memory_type);

private:
	///Maps every "name" to its Pin
  std::map<std::string, Pin> pins_;

	///An empty pin to return if a bad name is given
  Pin null_pin_;

	/**
		A specialized release function that can be set by the user.
	*/
	std::function<void()> prerelease_function_;

	///Depending on the situation, multiple writers may use this lock
	std::mutex write_lock_;
};

} //namespace sim

} //namespace ncs

#include <ncs/sim/DataBuffer.hpp>


