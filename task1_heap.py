import heapq


numbers = [12, 11, 13, 5, 6, 7]

heapq.heapify(numbers)

print("Heap:", numbers)

sorted_numbers = []

while numbers:
    sorted_numbers.append(heapq.heappop(numbers))

print("Sorted:", sorted_numbers)
